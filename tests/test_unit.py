"""Unit tests for the pure functions in proxy.py.

Expected values are pinned to the documented behaviour of each function.
"""

from proxy import (
    PrefixMemory,
    _hostport,
    attach_metrics,
    build_timings,
    cached_tokens_from_usage,
    content_to_text,
    delta_text,
    estimate_prompt_tokens,
    estimate_tokens,
    finish_reason_of,
    parse_serve_api_line,
    prefix_fingerprint,
    prompt_fingerprint,
    split_cached_prompt,
    usage_of,
)

SERVE_API_MTP = (
    "serve_api: mtp 250 tok in 6.75s = 37.05 t/s "
    "| 153 rounds, commit 1.63/round "
    "| prompt 1287 (1035 cached), prefill 1.26s"
)

SERVE_API_SERIAL = (
    "serve_api: serial 100 tok in 4.0s = 25.0 t/s | prompt 500, prefill 2.0s"
)

# halogen-flash-server >= 0.17: cached paren carries a percentage, prefill
# carries an explicit "(N new)".
SERVE_API_MTP_NEW = (
    "serve_api: mtp 57 tok in 0.90s = 63.42 t/s | 17 rounds, commit 3.29/round "
    "| prompt 185972 (185594 cached, 99.8%), prefill 0.66s (378 new) "
    "| detok 12us/tok | pld 10 rounds, 2.60 acc/round "
    "| pool 494592/524288 94% | think on"
)


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_whitespace_only(self):
        assert estimate_tokens("   \t\n ") == 0

    def test_latin_roughly_four_chars_per_token(self):
        assert estimate_tokens("hello world") == 2  # 10 non-space chars

    def test_single_char_is_at_least_one(self):
        assert estimate_tokens("a") == 1

    def test_cjk_counts_per_char_plus_slop(self):
        # Each CJK char counts as one token; the max(1, other//4) floor adds 1.
        assert estimate_tokens("你好世界") == 5

    def test_mixed(self):
        # cjk=2, other=8 -> max(1, 2 + max(1, 2)) = 4
        assert estimate_tokens("你好 hello world") == 4


class TestContentToText:
    def test_none(self):
        assert content_to_text(None) == ""

    def test_plain_string(self):
        assert content_to_text("hi") == "hi"

    def test_list_of_strings(self):
        assert content_to_text(["a", "b"]) == "ab"

    def test_typed_text_parts(self):
        parts = [
            {"type": "text", "text": "hello "},
            {"type": "image_url", "image_url": {"url": "x"}},
            {"type": "text", "text": "world"},
        ]
        assert content_to_text(parts) == "hello world"

    def test_untyped_dict_with_text(self):
        assert content_to_text([{"text": "x"}]) == "x"


class TestEstimatePromptTokens:
    def test_messages_have_template_slop(self):
        req = {"messages": [{"role": "user", "content": "hello world"}]}
        assert estimate_prompt_tokens(req) == 8 + 2

    def test_raw_prompt(self):
        assert estimate_prompt_tokens({"prompt": "hello world"}) == 8 + 2

    def test_reasoning_content_counts(self):
        bare = {"messages": [{"role": "assistant", "content": "hello world"}]}
        with_reasoning = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "hello world",
                    "reasoning_content": "hello world",
                }
            ]
        }
        assert estimate_prompt_tokens(with_reasoning) == (
            estimate_prompt_tokens(bare) + 2
        )

    def test_empty_request(self):
        assert estimate_prompt_tokens({}) == 8


class TestFingerprints:
    msgs = [{"role": "user", "content": "hi"}]

    def test_deterministic(self):
        a = prefix_fingerprint("m", self.msgs, 1)
        b = prefix_fingerprint("m", self.msgs, 1)
        assert a == b

    def test_model_sensitive(self):
        assert prefix_fingerprint("m1", self.msgs, 1) != prefix_fingerprint(
            "m2", self.msgs, 1
        )

    def test_prefix_length_matters(self):
        msgs = self.msgs + [{"role": "assistant", "content": "yo"}]
        assert prefix_fingerprint("m", msgs, 1) != prefix_fingerprint("m", msgs, 2)

    def test_prompt_fingerprint_distinct_from_messages(self):
        assert prompt_fingerprint("m", "hi") != prefix_fingerprint(
            "m", [{"role": "user", "content": "hi"}], 1
        )


class TestPrefixMemory:
    def test_remember_then_lookup(self):
        pm = PrefixMemory()
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        pm.remember(req, 100, 20, "yo")
        assert pm.lookup(req, 150) == 100

    def test_grown_history_hits_next_turn(self):
        pm = PrefixMemory()
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        pm.remember(req, 100, 20, "yo")
        grown = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ],
        }
        assert pm.lookup(grown, 150) == 120

    def test_model_mismatch_misses(self):
        pm = PrefixMemory()
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        pm.remember(req, 100, 20, "yo")
        other = {"model": "other", "messages": [{"role": "user", "content": "hi"}]}
        assert pm.lookup(other, 150) == 0

    def test_lookup_capped_at_total(self):
        pm = PrefixMemory()
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        pm.remember(req, 100, 20, "yo")
        # A remembered grown prefix may exceed the current prompt; it can
        # never claim more than the prompt itself.
        assert pm.lookup(req, 50) == 50

    def test_lru_eviction(self):
        pm = PrefixMemory(max_entries=2)
        for i in range(3):
            req = {"model": "m", "messages": [{"role": "user", "content": f"m{i}"}]}
            pm.remember(req, 100 + i, 1, "")
        oldest = {"model": "m", "messages": [{"role": "user", "content": "m0"}]}
        newest = {"model": "m", "messages": [{"role": "user", "content": "m2"}]}
        assert pm.lookup(oldest, 150) == 0
        assert pm.lookup(newest, 150) == 102

    def test_raw_prompt_roundtrip(self):
        pm = PrefixMemory()
        req = {"model": "m", "prompt": "hello world"}
        pm.remember(req, 50, 10, "gen")
        assert pm.lookup(req, 60) == 50
        # A grown prefix may cover the whole prompt; a full hit is allowed.
        assert pm.lookup({"model": "m", "prompt": "hello worldgen"}, 60) == 60


class TestParseServeApiLine:
    def test_mtp_line(self):
        row = parse_serve_api_line(SERVE_API_MTP)
        assert row is not None
        assert row["drafter"] == "mtp"
        assert row["predicted_n"] == 250
        assert row["decode_s"] == 6.75
        assert row["decode_tps"] == 37.05
        assert row["prompt_total"] == 1287
        assert row["cache_n"] == 1035
        assert row["prompt_n"] == 252
        assert row["prefill_s"] == 1.26
        assert row["prompt_tps"] == 200.0
        assert row["draft_n"] == 153
        assert row["draft_n_accepted"] == 97
        assert row["commit_per_round"] == 1.63

    def test_serial_line_has_no_draft(self):
        row = parse_serve_api_line(SERVE_API_SERIAL)
        assert row is not None
        assert row["drafter"] == "serial"
        assert row["predicted_n"] == 100
        assert row["prompt_total"] == 500
        assert row["cache_n"] == 0
        assert row["prompt_n"] == 500
        assert row["prompt_tps"] == 250.0
        assert row["draft_n"] == 0
        assert row["draft_n_accepted"] == 0

    def test_non_matching_line(self):
        assert parse_serve_api_line("serving something else entirely") is None

    def test_new_format_with_pct_and_new_count(self):
        row = parse_serve_api_line(SERVE_API_MTP_NEW)
        assert row is not None
        assert row["drafter"] == "mtp"
        assert row["predicted_n"] == 57
        assert row["decode_s"] == 0.90
        assert row["decode_tps"] == 63.42
        assert row["prompt_total"] == 185972
        assert row["cache_n"] == 185594
        assert row["prompt_n"] == 378
        assert row["prefill_s"] == 0.66
        assert abs(row["prompt_tps"] - 378 / 0.66) < 1e-6
        assert row["draft_n"] > 0
        assert row["draft_n_accepted"] == 57 - 17

    def test_explicit_new_count_beats_subtraction(self):
        line = (
            "serve_api: mtp 10 tok in 1.0s = 10.0 t/s "
            "| prompt 100 (90 cached), prefill 0.5s (7 new)"
        )
        row = parse_serve_api_line(line)
        assert row is not None
        assert row["prompt_n"] == 7

    def test_zero_prefill_gives_zero_rate(self):
        line = (
            "serve_api: serial 5 tok in 1.0s = 5.0 t/s "
            "| prompt 10 (10 cached), prefill 0.0s"
        )
        row = parse_serve_api_line(line)
        assert row is not None
        assert row["prompt_n"] == 0
        assert row["prompt_tps"] == 0.0


class TestSplitCachedPrompt:
    def test_no_cache_hint(self):
        assert split_cached_prompt(100, 0, 500.0) == (0, 100)

    def test_hint_respected_below_ceiling_window(self):
        # 100ms prefill is under the 150ms gate; no ceiling applied.
        assert split_cached_prompt(1000, 900, 100.0) == (900, 100)

    def test_ceiling_fires_on_impossible_rate(self):
        # 2000 fresh tokens in 500ms = 4000 tok/s > 1800 ceiling.
        cache_n, prompt_n = split_cached_prompt(2000, 0, 500.0)
        assert prompt_n == 900  # 1800 * 0.5
        assert cache_n == 1100

    def test_hint_capped_at_total(self):
        # A full hit means zero fresh prefill. The old prompt_n=1 floor charged
        # the whole TTFT to one token ("prompt=1.4 tok/s" on a warm 185k prompt).
        assert split_cached_prompt(10, 10, 500.0) == (10, 0)

    def test_full_hit_reports_zero_rate_not_one_token(self):
        t = build_timings(
            t0=0.0, t_first=0.7, t_end=2.0, prompt_n=0, predicted_n=57, cache_n=185594
        )
        assert t["prompt_n"] == 0
        assert t["prompt_per_second"] == 0.0
        assert t["prompt_per_token_ms"] == 0.0

    def test_zero_total(self):
        assert split_cached_prompt(0, 0, 500.0) == (0, 0)

    def test_custom_ceiling(self):
        cache_n, prompt_n = split_cached_prompt(2000, 0, 500.0, ceiling=1000)
        assert prompt_n == 500
        assert cache_n == 1500


class TestBuildTimings:
    def test_normal(self):
        t = build_timings(t0=0.0, t_first=1.0, t_end=3.0, prompt_n=100, predicted_n=50)
        assert t["cache_n"] == 0
        assert t["prompt_n"] == 100
        assert t["prompt_ms"] == 1000.0
        assert t["prompt_per_second"] == 100.0
        assert t["prompt_per_token_ms"] == 10.0
        assert t["predicted_n"] == 50
        assert t["predicted_ms"] == 2000.0
        assert t["predicted_per_second"] == 25.0
        assert t["predicted_per_token_ms"] == 40.0

    def test_no_first_token(self):
        t = build_timings(t0=1.0, t_first=None, t_end=1.0, prompt_n=10, predicted_n=0)
        assert t["prompt_ms"] == 0.0
        assert t["prompt_per_second"] == 0.0
        assert t["predicted_ms"] == 0.0
        assert t["predicted_per_second"] == 0.0

    def test_negative_deltas_clamped(self):
        t = build_timings(t0=5.0, t_first=1.0, t_end=0.0, prompt_n=10, predicted_n=10)
        assert t["prompt_ms"] == 0.0
        assert t["predicted_ms"] == 0.0

    def test_zero_counts_no_division_error(self):
        t = build_timings(t0=0.0, t_first=1.0, t_end=2.0, prompt_n=0, predicted_n=0)
        assert t["prompt_per_token_ms"] == 0.0
        assert t["predicted_per_token_ms"] == 0.0


class TestAttachMetrics:
    def test_adds_timings_and_synthesizes_usage(self):
        obj = {"choices": [{"finish_reason": "stop"}]}
        timings = build_timings(
            t0=0.0, t_first=1.0, t_end=2.0, prompt_n=100, predicted_n=50
        )
        out = attach_metrics(obj, timings, None)
        assert out["timings"] is timings
        assert out["usage"] == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    def test_does_not_clobber_real_upstream_timings(self):
        real = {"predicted_per_second": 42.0}
        obj = {"timings": real}
        forged = {
            "predicted_per_second": 1.0,
            "prompt_n": 5,
            "predicted_n": 5,
        }
        attach_metrics(obj, forged, None)
        assert obj["timings"] is real

    def test_explicit_usage_wins(self):
        obj = {"usage": {"prompt_tokens": 1}}
        usage = {"prompt_tokens": 9, "completion_tokens": 9, "total_tokens": 18}
        attach_metrics(obj, {"prompt_n": 0, "predicted_n": 0}, usage)
        assert obj["usage"] == usage


class TestHostport:
    def test_plain_host_port(self):
        assert _hostport("127.0.0.1:1234", default_port=8731) == (
            "127.0.0.1",
            1234,
        )

    def test_url_form(self):
        assert _hostport("http://api:8731", default_port=8731) == ("api", 8731)

    def test_bare_host_gets_default(self):
        assert _hostport("api", default_port=8731) == ("api", 8731)

    def test_ipv6_brackets(self):
        assert _hostport("[::1]:9999", default_port=8731) == ("::1", 9999)


class TestSmallHelpers:
    def test_finish_reason_of(self):
        assert finish_reason_of({"choices": [{"finish_reason": "stop"}]}) == "stop"
        assert finish_reason_of({"choices": [{"delta": {}}]}) is None
        assert finish_reason_of({}) is None

    def test_usage_of(self):
        assert usage_of({"usage": {"a": 1}}) == {"a": 1}
        assert usage_of({"usage": "nope"}) is None
        assert usage_of({}) is None

    def test_delta_text_joins_content_and_reasoning(self):
        assert delta_text({"content": "a", "reasoning_content": "b"}) == "ab"
        assert delta_text({}) == ""
        assert delta_text("not a dict") == ""

    def test_cached_tokens_from_usage(self):
        assert cached_tokens_from_usage({"cached_tokens": "5"}) == 5
        assert cached_tokens_from_usage({"cache_tokens": 3}) == 3
        assert (
            cached_tokens_from_usage({"prompt_tokens_details": {"cached_tokens": 7}})
            == 7
        )
        assert cached_tokens_from_usage({"cached_tokens": "bad"}) == 0
        assert cached_tokens_from_usage(None) == 0
