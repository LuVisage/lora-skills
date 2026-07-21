# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 2.4.x   | ✅ Actively supported |
| 2.3.x   | ✅ Security fixes only |
| < 2.3.0 | ❌ End of life      |

## Reporting a Vulnerability

**Do not open a public issue.** If you discover a security vulnerability, please report it privately via GitHub's Security Advisory feature:

1. Go to https://github.com/LuVisage/lora-skills/security/advisories/new
2. Describe the vulnerability in detail
3. Include steps to reproduce if possible

You will receive a response within 48 hours. We will coordinate the fix and disclosure timeline with you.

## Security Considerations for lora-trainer

### Generated Training Scripts

The `cook` and `evaluate` commands generate Python scripts. If you distribute generated scripts:

- Review `MODEL_NAME` and `DATA_PATH` — never hardcode absolute paths with sensitive info
- Generated scripts import from PyPI packages — verify your dependency supply chain

### Model Security

- Only fine-tune models from trusted sources (Hugging Face verified orgs, official repositories)
- Be aware that LoRA adapters can carry the same supply-chain risks as full model weights
- `trust_remote_code=True` is set in generated scripts — only use with models you trust
