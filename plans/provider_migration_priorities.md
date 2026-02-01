# RAGFlow LLM Provider Migration Priorities to LiteLLM

## Current Provider Landscape Analysis

Based on codebase analysis, here are all LLM provider implementations found in `rag/llm/chat_models/`:

### 1. Already Using LiteLLM (33 providers)
These providers are already implemented via `LiteLLMBase` class:
- **Tongyi-Qianwen** (Dashscope)
- **Bedrock** (with IAM role/auth caching)
- **Moonshot**
- **xAI**
- **DeepInfra**
- **Groq**
- **Cohere**
- **Gemini**
- **DeepSeek**
- **NVIDIA**
- **TogetherAI**
- **Anthropic**
- **Ollama** (with Bearer token support)
- **LongCat**
- **CometAPI**
- **SILICONFLOW**
- **OpenRouter** (with provider ordering)
- **StepFun**
- **PPIO**
- **PerfXCloud**
- **Upstage**
- **NovitaAI**
- **01.AI** (Lingyi)
- **GiteeAI**
- **302.AI**
- **Jiekou.AI**
- **ZHIPU-AI**
- **MiniMax**
- **DeerAPI**
- **GPUStack**
- **OpenAI**
- **Azure-OpenAI** (with api_version support)
- **Tencent Hunyuan**

**Status:** These are already migrated to LiteLLM - no further action needed.

### 2. Custom Provider Implementations (To Be Assessed)
These have custom implementations that may need migration:

| Provider | File | Complexity | Migration Priority |
|----------|------|------------|-------------------|
| **BaiduYiyan** | `baidu.py` | Medium | Medium |
| **Google** | `google.py` | Low | High |

| **Spark** (XunFei) | `spark.py` | Low | High |
| **VolcEngine** | `volcengine.py` | Low | High |
| **Mistral** | `mistral.py` | Low | High |
| **LeptonAI** | `lepton.py` | Low | High |
| **Replicate** | `replicate.py` | Low | High |
| **TokenPony** | `tokenpony.py` | Low | High |
| **Xinference** | `xinference.py` | Low | High |
| **HuggingFace** | `huggingface.py` | High | Low |
| **ModelScope** | `modelscope.py` | High | Low |
| **BaiChuan** | `baichuan.py` | Medium | Medium |
| **LocalAI** | `localai.py` | Low | High |
| **LmStudio** | `lmstudio.py` | Low | High |

| **LocalLLM** | `localllm.py` | High | Low |

## Migration Priority Assessment Matrix

### Criteria for Assessment:
1. **Complexity**: How complex is the current implementation?
2. **LiteLLM Support**: Does LiteLLM support this provider natively?
3. **Usage Frequency**: How often is this provider used in RAGFlow deployments?
4. **Feature Parity**: Can LiteLLM provide the same features?
5. **Breaking Changes Risk**: Risk of breaking existing deployments.

### Priority Tiers:

#### Tier 1: Easy Wins (High Priority)
These providers have simple implementations and are likely fully supported by LiteLLM:
- **Google** - Likely supported via `gemini/` or `google/` provider
- **Spark (XunFei)** - May be supported via Chinese provider mappings
- **VolcEngine** - Likely supported as OpenAI-compatible
- **Mistral** - Supported via `mistral/` or `openai/` provider
- **LeptonAI** - OpenAI-compatible
- **Replicate** - Likely supported
- **TokenPony** - Status TBD: Needs investigation to confirm LiteLLM support and migration complexity (move to Tier 1 once confirmed)
- **Xinference** - OpenAI-compatible
- **LocalAI** - Already supported in LiteLLM (should be using LiteLLM)
- **LmStudio** - Already supported in LiteLLM (should be using LiteLLM)

**Action:** Migrate these first as they provide quick wins with minimal risk.

#### Tier 2: Medium Complexity (Medium Priority)
These have some custom logic but could potentially use LiteLLM:
- **BaiduYiyan** - Uses Qianfan SDK, LiteLLM may have `baidu/` provider

- **BaiChuan** - Chinese provider, needs investigation
- **OpenAI_APIChat** - Already covered by LiteLLM's OpenAI provider

**Action:** Investigate LiteLLM support, create compatibility layer if needed.

#### Tier 3: High Complexity (Low Priority)
These have complex implementations or may not be well-supported by LiteLLM:
- **HuggingFace** - Complex local inference, custom endpoints
- **ModelScope** - Chinese ModelScope integration
- **LocalLLM** - Generic local LLM wrapper, highly customizable

**Action:** Maintain custom implementations, consider contributing to LiteLLM.

## Phase 1 Migration Plan (0-3 months)

### Week 1-2: Foundation
1. **Audit & Testing Framework**
   - Create provider compatibility test suite
   - Benchmark current vs LiteLLM performance
   - Document authentication requirements for each provider

2. **Configuration Standardization**
   - Standardize JSON configuration schema
   - Create migration utility for existing deployments

3. **Operational Readiness**
   - **Documentation**: Create 4 migration guides (Google, Spark, VolcEngine, Mistral) and API docs for LiteLLM integration
   - **User Communication**: Schedule deprecation notices and migration timelines
   - **CI/CD**: Update test suites/deployment pipelines (enable `USE_LITELLM` flag tests) and run integration tests on config migration
   - **Versioning Strategy**: Define semantic version bumps and compatibility matrix for config -> LiteLLM expected format

4. **TokenPony Investigation**
   - Owner: TBD | Timeline: 3-5 days
   - Acceptance Criteria: Verify LiteLLM compatibility, estimate complexity.
   - Outcome: Update status to Tier 1 or document blocker; schedule migration.

### Week 3-6: Tier 1 Providers (Extended Schedule)
1. **Google → LiteLLM migration** (Week 3)
2. **Spark (XunFei) → LiteLLM migration** (Week 4)
3. **VolcEngine → LiteLLM migration** (Week 5)
4. **Mistral → LiteLLM migration** (Week 6)
5. **Additional Tier 1 Migrations** (Month 2)
   - LeptonAI, Replicate, Xinference, LocalAI, LmStudio (Deferred to Month 2 to prioritize stability of initial batch; see Tier 1 list)
   - TokenPony (Conditional on investigation results)

**Success Criteria:** All tests pass, no breaking changes for existing users. (Includes 20–30% contingency buffer for unit/integration/perf test cycles and rollback validation)

### Month 2: Tier 2 Providers
1. **BaiduYiyan → LiteLLM investigation**
2. **Hunyuan → LiteLLM migration** (if supported)
3. **BaiChuan → LiteLLM investigation**
4. **LocalAI/LmStudio consolidation** (ensure they use LiteLLM)

### Month 3: Assessment & Planning
1. **Performance comparison report**
2. **User feedback collection**
3. **Tier 3 provider roadmap decision**

## Technical Migration Approach

### For Each Provider:
1. **Analysis Phase:**
   - Check LiteLLM provider support: `litellm.supports_provider(provider_name)`
   - Compare authentication requirements
   - Test basic chat completion

2. **Wrapper Development:**
   - Create `LiteLLMBase` subclass if needed
   - Handle provider-specific configuration parsing
   - Maintain backward compatibility

3. **Testing:**
   - Unit tests for authentication
   - Integration tests for chat completion
   - Performance benchmarks
   - Error handling verification

### Configuration Migration:
Current provider configs need to map to LiteLLM's expected format:

```python
# Example: VolcEngine current config
{
  "ark_api_key": "key",
  "ep_id": "id",
  "endpoint_id": "endpoint"
}

# LiteLLM expected format
{
  "api_key": "key",
  "model": "volcengine/{ep_id}-{endpoint_id}",
  "api_base": "https://ark.cn-beijing.volces.com/api/v3"
}
```

## Risk Mitigation Strategies

### 1. Backward Compatibility
- Maintain both implementations during transition
- Feature flag to switch between implementations
- Configuration auto-detection and migration

### 2. Fallback Mechanism
```python
def get_provider(provider_name, config):
    if USE_LITELLM.get(provider_name, False):
        try:
            return LiteLLMProvider(config)
        except Exception as e:
            logging.warning(f"LiteLLM failed, falling back: {e}")
    
    return LegacyProvider(config)
```

### 3. Monitoring & Rollback
- Detailed logging of provider performance
- Automatic rollback if error rates exceed threshold
- A/B testing framework for comparison

## Success Metrics

1. **Code Reduction:** Target 50% reduction in provider-specific code
2. **Maintenance Burden:** Reduced issues related to provider updates
3. **Performance:** No significant latency increase (<10%)
4. **Compatibility:** 100% backward compatibility for existing configurations
5. **New Provider Onboarding:** Reduced from days to hours

## Next Steps

1. **Immediate:**
   - Audit current provider usage statistics
   - Test LiteLLM support for Tier 1 providers
   - Create migration test suite

2. **Short-term:**
   - Implement first provider migration (Google)
   - Gather user feedback
   - Refine migration approach

3. **Long-term:**
   - Full migration assessment after 3 providers
   - Decision on Tier 3 providers
   - Contribution plan for LiteLLM gaps

## Appendix: Provider-Specific Notes

### Google
- Current: Uses `google.generativeai` SDK
- LiteLLM: Likely `gemini/` provider
- Risk: Low, Gemini API is well-supported

### Spark (XunFei)
- Current: Custom version mapping
- LiteLLM: Check `spark/` or Chinese provider support
- Risk: Medium, may need custom adapter

### VolcEngine
- Current: JSON config parsing
- LiteLLM: Likely OpenAI-compatible
- Risk: Low

### BaiduYiyan
- Current: Qianfan SDK with AK/SK
- LiteLLM: May need `baidu/` provider or contribute support
- Risk: Medium

### HuggingFace & ModelScope
- Current: Complex local inference
- LiteLLM: May not support all features
- Recommendation: Keep custom implementation