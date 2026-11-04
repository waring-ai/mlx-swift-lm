// Copyright © 2025 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Unit tests for the protocol-agnostic cache-reuse rules that `ChatSession`
/// applies at the start of every turn. The policy is pure, so the whole
/// decision table is exercised without a model, a tokenizer, or a KV cache.
@Suite
struct PromptCacheReusePolicyTests {

    // MARK: - Fixtures

    private func turn(
        prompt: [Int],
        newMedia: Bool = false,
        preparedMedia: Bool = false,
        attentionMask: Bool = false,
        modelState: Bool = false,
        toolResultContinuation: Bool = false,
        speculative: Bool = false,
        canSplitMedia: Bool = false
    ) -> PromptCacheTurn {
        PromptCacheTurn(
            promptTokens: prompt,
            carriesNewMedia: newMedia,
            carriesPreparedMedia: preparedMedia,
            carriesAttentionMask: attentionMask,
            carriesModelState: modelState,
            isToolResultContinuation: toolResultContinuation,
            usesSpeculativeDecoding: speculative,
            canSplitPreparedMedia: canSplitMedia)
    }

    /// A cache whose model-wide timeline agrees with `cached`, unless
    /// `processed` overrides it.
    private func alignedCache(
        _ cached: [Int],
        processed: Int? = nil,
        speculativeReuseCapability: SpeculativeCacheReuseCapability = .unavailable,
        trimmable: Bool = true
    ) -> PromptCacheState {
        let processedTokenCount = processed ?? cached.count
        return PromptCacheState(
            cachedTokens: cached,
            processedTokenCount: processedTokenCount,
            mainCacheIsAligned: processedTokenCount == cached.count,
            speculativeReuseCapability: speculativeReuseCapability,
            maxTrimCount: trimmable ? cached.count : 0)
    }

    // MARK: - Fresh and empty caches

    @Test func `an empty cache prefills the whole prompt`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 3]), cache: alignedCache([]))

        #expect(decision == .prefillAll)
        #expect(decision.reusesCachedPrefix == false)
    }

    @Test func `a populated cache with an invalidated ledger rebuilds`() {
        // A cancelled generation clears the ledger but leaves tokens in the
        // cache: there is nothing to splice onto and the cache is stale.
        let cache = PromptCacheState(
            cachedTokens: [],
            processedTokenCount: 12,
            mainCacheIsAligned: false,
            maxTrimCount: 12)

        #expect(
            PromptCacheReusePolicy().decide(turn: turn(prompt: [1, 2, 3]), cache: cache) == .rebuild
        )
    }

    // MARK: - Prefix extension

    @Test func `a prompt extending the cache appends only its suffix`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 3, 4, 5]), cache: alignedCache([1, 2, 3]))

        #expect(decision == .appendSuffix(suffixStart: 3, representedTokens: [1, 2, 3, 4, 5]))
        #expect(decision.reusesCachedPrefix)
    }

    @Test func `an identical prompt rewinds one token to refeed its last token`() {
        // No new tokens: the common prefix spans the whole prompt, so the last
        // token is rewound and fed again to produce logits.
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(prompt: [1, 2, 3]), cache: alignedCache([1, 2, 3]))
                == .trimToCommonPrefix(commonPrefixLength: 2, trimCount: 1))
    }

    @Test func `a prompt the cache extends past rewinds to just before its last token`() {
        // A retry of a turn whose response is cached: the prompt is a strict
        // prefix of the ledger, which also holds the generated tokens.
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(prompt: [1, 2, 3]), cache: alignedCache([1, 2, 3, 4, 5]))
                == .trimToCommonPrefix(commonPrefixLength: 2, trimCount: 3))
    }

    @Test func `a single-token prompt the cache holds rebuilds`() {
        // Rewinding one token would leave nothing to reuse.
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(prompt: [1]), cache: alignedCache([1, 2])) == .rebuild)
    }

    @Test func `a misaligned draft cache blocks suffix reuse`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 3, 4], speculative: true),
            cache: alignedCache([1, 2, 3], speculativeReuseCapability: .unavailable))

        #expect(decision == .rebuild)
    }

    @Test func `a missing ordinary draft remains tentatively rebuildable`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 3, 4], speculative: true),
            cache: alignedCache(
                [1, 2, 3], speculativeReuseCapability: .rebuildableFromPrompt))

        #expect(decision == .appendSuffix(suffixStart: 3, representedTokens: [1, 2, 3, 4]))
    }

    @Test func `new media invalidates the cached text prefix`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 3, 4], newMedia: true), cache: alignedCache([1, 2, 3]))

        #expect(decision == .rebuild)
    }

    // MARK: - Append-only media turns

    @Test func `an append-only media turn appends its suffix when the model can split`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 3, 4], newMedia: true, canSplitMedia: true),
            cache: alignedCache([1, 2, 3]))

        #expect(
            decision == .appendMediaSuffix(suffixStart: 3, representedTokens: [1, 2, 3, 4]))
        #expect(decision.reusesCachedPrefix)
    }

    /// A VL processor pairs media with an all-ones mask as a matter of course, so
    /// the media path must not inherit ``ExtendCachedPrefixRule``'s mask guard --
    /// doing so would disable it for exactly the inputs it serves. Telling a
    /// benign mask from a real padding mask needs the model's layout knowledge,
    /// so that check belongs to the splitter.
    @Test func `a benign attention mask does not block the media split`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(
                prompt: [1, 2, 3, 4], newMedia: true, attentionMask: true, canSplitMedia: true),
            cache: alignedCache([1, 2, 3]))

        #expect(
            decision == .appendMediaSuffix(suffixStart: 3, representedTokens: [1, 2, 3, 4]))
    }

    @Test func `a model that cannot split keeps the rebuild on a media turn`() {
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(prompt: [1, 2, 3, 4], newMedia: true, canSplitMedia: false),
                cache: alignedCache([1, 2, 3])) == .rebuild)
    }

    @Test func `speculative decoding blocks the media split`() {
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(
                    prompt: [1, 2, 3, 4], newMedia: true, speculative: true,
                    canSplitMedia: true),
                cache: alignedCache([1, 2, 3])) == .rebuild)
    }

    @Test func `a media turn that diverges from the cache does not split`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 9, 9], newMedia: true, canSplitMedia: true),
            cache: alignedCache([1, 2, 3]))

        #expect(decision == .rebuild)
    }

    @Test func `a misaligned draft cache blocks the media split`() {
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(
                    prompt: [1, 2, 3, 4], newMedia: true, speculative: true,
                    canSplitMedia: true),
                cache: alignedCache([1, 2, 3], speculativeReuseCapability: .unavailable))
                == .rebuild)
    }

    @Test func `an explicit attention mask blocks suffix reuse`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 3, 4], attentionMask: true), cache: alignedCache([1, 2, 3]))

        #expect(decision == .rebuild)
    }

    @Test func `aligned resumable MTP state appends only the suffix`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 3, 4], speculative: true),
            cache: alignedCache([1, 2, 3], speculativeReuseCapability: .appendOnly))

        #expect(decision == .appendSuffix(suffixStart: 3, representedTokens: [1, 2, 3, 4]))
    }

    @Test func `missing MTP continuation preserves the target prefix target-only`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 3, 4], speculative: true),
            cache: alignedCache([1, 2, 3], speculativeReuseCapability: .mainOnlyFallback))

        #expect(
            decision
                == .appendSuffixToMain(
                    suffixStart: 3, representedTokens: [1, 2, 3, 4]))
    }

    // MARK: - Rewind to the common prefix

    @Test func `a divergent prompt rewinds to the common prefix`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 9, 9]), cache: alignedCache([1, 2, 3, 4, 5]))

        #expect(decision == .trimToCommonPrefix(commonPrefixLength: 2, trimCount: 3))
        #expect(decision.reusesCachedPrefix)
    }

    /// A cache that can rewind only `maxTrimCount` tokens, with a prompt checkpoint.
    private func checkpointedCache(
        _ cached: [Int], maxTrimCount: Int, checkpoint: Int?
    ) -> PromptCacheState {
        PromptCacheState(
            cachedTokens: cached, processedTokenCount: cached.count,
            mainCacheIsAligned: true, speculativeReuseCapability: .reusable,
            maxTrimCount: maxTrimCount, checkpointTokenCount: checkpoint)
    }

    @Test func `a rewind past the reserve restores the checkpoint within the common prefix`() {
        // The cache holds prompt [1, 2, 3] and response [4, 5, 6]; the next turn
        // drops the response, which the cache can rewind only one token over.
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 3, 9, 9]),
            cache: checkpointedCache([1, 2, 3, 4, 5, 6], maxTrimCount: 1, checkpoint: 3))

        #expect(decision == .restoreCheckpoint(tokenCount: 3))
        #expect(decision.reusesCachedPrefix)
    }

    @Test func `a rewind within the reserve trims rather than restoring the checkpoint`() {
        // Trimming keeps more of the cache than the checkpoint does.
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(prompt: [1, 2, 3, 4, 5, 9]),
                cache: checkpointedCache([1, 2, 3, 4, 5, 6], maxTrimCount: 1, checkpoint: 3))
                == .trimToCommonPrefix(commonPrefixLength: 5, trimCount: 1))
    }

    @Test func `a checkpoint past the common prefix is not restored`() {
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(prompt: [1, 2, 9, 9]),
                cache: checkpointedCache([1, 2, 3, 4, 5, 6], maxTrimCount: 1, checkpoint: 3))
                == .rebuild)
    }

    @Test func `a checkpoint covering the whole prompt is not restored`() {
        // Its last token must still be fed, which the checkpoint has already consumed.
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(prompt: [1, 2, 3]),
                cache: checkpointedCache([1, 2, 3, 4, 5, 6], maxTrimCount: 1, checkpoint: 3))
                == .rebuild)
    }

    @Test func `speculative decoding blocks restoring the main cache's checkpoint`() {
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(prompt: [1, 2, 3, 9], speculative: true),
                cache: checkpointedCache([1, 2, 3, 4, 5, 6], maxTrimCount: 1, checkpoint: 3))
                == .rebuild)
    }

    @Test func `carried model state blocks restoring a checkpoint`() {
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(prompt: [1, 2, 3, 9], modelState: true),
                cache: checkpointedCache([1, 2, 3, 4, 5, 6], maxTrimCount: 1, checkpoint: 3))
                == .rebuild)
    }

    @Test func `a fully divergent prompt rebuilds rather than rewinding to nothing`() {
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(prompt: [9, 9, 9]), cache: alignedCache([1, 2, 3])) == .rebuild)
    }

    @Test func `stateful MTP rewind keeps only the authoritative target cache`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 9, 9], speculative: true),
            cache: alignedCache(
                [1, 2, 3, 4, 5], speculativeReuseCapability: .appendOnly))

        #expect(
            decision
                == .trimToCommonPrefixMainOnly(
                    commonPrefixLength: 2, trimCount: 3))
    }

    @Test func `stateless MTP can follow a target rewind`() {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(prompt: [1, 2, 9, 9], speculative: true),
            cache: alignedCache([1, 2, 3, 4, 5], speculativeReuseCapability: .reusable))

        #expect(decision == .trimToCommonPrefix(commonPrefixLength: 2, trimCount: 3))
    }

    @Test(arguments: [
        SpeculativeCacheReuseCapability.reusable, .appendOnly,
        .rebuildableFromPrompt, .mainOnlyFallback,
    ])
    func `speculative rewinds respect the exact cache rewind limit`(
        capability: SpeculativeCacheReuseCapability
    ) {
        var cache = alignedCache([1, 2, 3, 4, 5], speculativeReuseCapability: capability)
        cache.maxTrimCount = 2

        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(prompt: [1, 2, 9, 9], speculative: true), cache: cache) == .rebuild)
    }

    @Test(arguments: [
        "prepared media", "attention mask", "model state", "not trimmable",
    ])
    func `a rewind is refused when it cannot be performed safely`(blocker: String) {
        let decision = PromptCacheReusePolicy().decide(
            turn: turn(
                prompt: [1, 2, 9, 9],
                preparedMedia: blocker == "prepared media",
                attentionMask: blocker == "attention mask",
                modelState: blocker == "model state"),
            cache: alignedCache([1, 2, 3, 4, 5], trimmable: blocker != "not trimmable"))

        #expect(decision == .rebuild)
    }

    // MARK: - Rule composition

    @Test func `a protocol rule is consulted before the standard rules`() {
        // A response protocol whose token stream is not reproducible by the
        // chat template must be able to claim a turn that the standard rules
        // would otherwise resolve by prefix comparison.
        struct ClaimEverythingRule: PromptCacheReuseRule {
            func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision?
            {
                .appendSuffix(suffixStart: 0, representedTokens: [42])
            }
        }

        let policy = PromptCacheReusePolicy(protocolRules: [ClaimEverythingRule()])

        #expect(
            policy.decide(turn: turn(prompt: [1, 2, 3, 4]), cache: alignedCache([1, 2, 3]))
                == .appendSuffix(suffixStart: 0, representedTokens: [42]))
    }

    @Test func `a declining protocol rule leaves the standard behavior intact`() {
        struct DecliningRule: PromptCacheReuseRule {
            func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision?
            {
                nil
            }
        }

        let withRule = PromptCacheReusePolicy(protocolRules: [DecliningRule()])
        let standard = PromptCacheReusePolicy()
        let turn = turn(prompt: [1, 2, 3, 4])
        let cache = alignedCache([1, 2, 3])

        #expect(
            withRule.decide(turn: turn, cache: cache) == standard.decide(turn: turn, cache: cache))
    }
}
