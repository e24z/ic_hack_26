"""Verification script for the configuration system."""

import asyncio
import os

from config.loader import load_config
from config.factory import create_from_profile


async def verify_profile(profile_name: str):
    """Verify a specific profile can be loaded and creates backends."""
    print(f"\n{'=' * 60}")
    print(f"Verifying profile: {profile_name}")
    print(f"{'=' * 60}")

    try:
        # Load profile
        profile = load_config(profile=profile_name)
        print(f"✓ Config loaded successfully")
        print(f"  - Summarizer: {profile.summarizer.backend}")
        print(f"  - NLI: {profile.nli.backend}")
        print(f"  - Lettuce: {profile.lettuce.backend}")

        # Create backends
        summarizer, nli_backend, validator = create_from_profile(profile)
        print(f"✓ Backends created successfully")
        print(f"  - Summarizer type: {type(summarizer).__name__}")
        print(f"  - NLI type: {type(nli_backend).__name__}")
        print(f"  - Validator type: {type(validator).__name__}")

        return True

    except Exception as exc:
        print(f"❌ Error: {exc}")
        return False


async def main():
    """Run verification tests."""
    print("Configuration System Verification")
    print("=" * 60)

    profiles = ["dev-fast", "dev-accurate", "test", "prod", "local-http"]
    results = {}

    for profile_name in profiles:
        results[profile_name] = await verify_profile(profile_name)

    # Summary
    print(f"\n{'=' * 60}")
    print("Verification Summary")
    print(f"{'=' * 60}")

    for profile_name, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status}: {profile_name}")

    # Test env var expansion
    print(f"\n{'=' * 60}")
    print("Testing Environment Variable Expansion")
    print(f"{'=' * 60}")

    profile = load_config(profile="dev-fast")
    if profile.summarizer.api_key:
        print("✓ API key expanded from environment variable")
        print(f"  Key length: {len(profile.summarizer.api_key)} characters")
    else:
        print("❌ API key not found")

    # Test backward compatibility
    print(f"\n{'=' * 60}")
    print("Testing Backward Compatibility (.env fallback)")
    print(f"{'=' * 60}")

    import shutil
    from pathlib import Path

    config_path = Path("config/models.yaml")
    backup_path = Path("config/models.yaml.backup")

    if config_path.exists():
        shutil.move(config_path, backup_path)
        try:
            profile = load_config()
            print("✓ Fallback to .env works")
            print(f"  - Summarizer backend: {profile.summarizer.backend}")
            print(f"  - NLI backend: {profile.nli.backend}")
            print(f"  - Lettuce backend: {profile.lettuce.backend}")
        except Exception as exc:
            print(f"❌ Fallback failed: {exc}")
        finally:
            shutil.move(backup_path, config_path)


if __name__ == "__main__":
    asyncio.run(main())
