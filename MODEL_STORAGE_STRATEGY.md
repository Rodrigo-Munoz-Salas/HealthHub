# Model Storage Strategy

## Current Issue
- TinyLlama model: 1.1GB (exceeds GitHub's 100MB limit)
- Total models: 1.2GB (makes repo very large)

## Recommended Solution: Hugging Face Hub
1. Create account at https://huggingface.co
2. Create a new model repository
3. Upload models using git or web interface
4. Reference models in your code using Hugging Face's API

## Alternative: GitHub Releases
1. Create a GitHub release
2. Attach model files as release assets
3. Download models programmatically in your app

## Current Status
Models are committed to 'new-branch' but will cause issues if pushed to GitHub.
