from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from config.loader import load_config
from config.factory import create_from_profile

# Load environment variables from .env file
load_dotenv()


async def test_good_summary(summarizer, validator) -> None:
    """Test with a correct summary."""
    print("=" * 60)
    print("TEST 1: Good Summary (Should Pass)")
    print("=" * 60)

    context = "The Eiffel Tower is located in Paris, France. It was completed in 1889."

    # 1. Summarize
    print("\n[1/3] Generating summary...")
    summary = await summarizer.generate_summary(
        data=context, guidance="Create a brief summary."
    )
    print(f"Summary: {summary}")

    # 2. Validate
    print("\n[2/3] Validating with LettuceDetect + NLI...")
    result = await validator.validate(summary=summary, context=context)

    print("\n[3/3] Validation Results:")
    print(f"  Grounded: {result.grounded_pct:.1%}")
    print(f"  Blocked: {result.blocked}")
    print(f"  Spans detected: {len(result.spans)}")
    for span in result.spans:
        print(
            f"    - '{span.text}' [{span.label}] (confidence: {span.confidence:.2f}, hallucination_score: {span.hallucination_score:.2f})"
        )

    if result.blocked:
        print("\n❌ TEST FAILED: Good summary was blocked!")
    else:
        print("\n✓ TEST PASSED: Good summary was not blocked")


async def test_bad_summary(validator) -> None:
    """Test with a hallucinated summary."""
    print("\n\n" + "=" * 60)
    print("TEST 2: Bad Summary (Should Be Blocked)")
    print("=" * 60)

    context = "The Eiffel Tower is located in Paris, France. It was completed in 1889."
    bad_summary = "The Eiffel Tower is located in London, England. It was completed in 1889."

    print(f"\nContext: {context}")
    print(f"Bad Summary: {bad_summary}")

    # Validate
    print("\n[1/2] Validating with LettuceDetect + NLI...")
    result = await validator.validate(summary=bad_summary, context=context)

    print("\n[2/2] Validation Results:")
    print(f"  Grounded: {result.grounded_pct:.1%}")
    print(f"  Blocked: {result.blocked}")
    print(f"  Spans detected: {len(result.spans)}")
    for span in result.spans:
        print(
            f"    - '{span.text}' [{span.label}] (confidence: {span.confidence:.2f}, hallucination_score: {span.hallucination_score:.2f})"
        )

    if result.blocked:
        print("\n✓ TEST PASSED: Bad summary was correctly blocked")
    else:
        print("\n❌ TEST FAILED: Bad summary was not blocked!")


async def main() -> None:
    """Run all tests using configuration system."""
    print("Loading configuration...")

    # Load configuration from YAML or env vars
    try:
        profile = load_config()
    except Exception as exc:
        print(f"❌ Failed to load configuration: {exc}")
        print("\nMake sure you have:")
        print("1. Created config/models.yaml (or it will use .env fallback)")
        print("2. Set MODEL_PROFILE env var (default: dev-fast)")
        print("3. Set OPENROUTER_API_KEY in .env file")
        return

    # Display active profile
    profile_name = os.environ.get("MODEL_PROFILE", "dev-fast")
    print(f"Using profile: {profile_name}")
    print(f"  - Summarizer: {profile.summarizer.backend}")
    print(f"  - NLI: {profile.nli.backend}")
    print(f"  - Lettuce: {profile.lettuce.backend}")
    print()

    # Create backends from profile
    try:
        summarizer, nli_backend, validator = create_from_profile(profile)
    except Exception as exc:
        print(f"❌ Failed to create backends: {exc}")
        return

    try:
        await test_good_summary(summarizer, validator)
        await test_bad_summary(validator)
    except Exception as exc:
        print(f"\n❌ ERROR: {exc}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
