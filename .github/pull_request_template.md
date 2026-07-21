## Description

<!-- What does this PR do? -->

## Type of Change

- [ ] 🐛 Bug fix
- [ ] ✨ New feature
- [ ] 📚 Documentation
- [ ] 🔧 Configuration / infra
- [ ] 🎨 Style / refactor

## Checklist

- [ ] Smoke test passes: `python -c "from scripts.analyzer import quick_analyze; from scripts.memory_calc import quick_calc; from scripts.lora_advisor import quick_recommend; from cli.main import main; print('✅ OK')"`
- [ ] CLI commands tested: `python -m cli.main analyze examples/sample_data.jsonl`
- [ ] Version bumped in all 4 places (if applicable): `pyproject.toml`, `package.json`, `.claude-plugin/plugin.json`, `cli/__init__.py`
- [ ] CHANGELOG.md updated (if applicable)
- [ ] No changes to `scripts/*.py` unless fixing a computation bug
