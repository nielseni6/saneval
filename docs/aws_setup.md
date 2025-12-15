# AWS Bedrock Setup for SANEval

This guide explains how to configure AWS credentials and access for using Llama 4 models via AWS Bedrock in SANEval.

## Prerequisites

### 1. AWS Account
- You need an active AWS account
- Sign up at https://aws.amazon.com/ if you don't have one

### 2. AWS Bedrock Access
- AWS Bedrock must be enabled in your account
- Llama 4 model access must be requested and approved

### 3. Python Dependencies
- boto3 and botocore are installed (included in requirements.txt)

```bash
pip install -r requirements.txt
```

## Requesting Bedrock Model Access

Before you can use Llama 4 models, you must request access:

1. Log into the AWS Console
2. Navigate to **AWS Bedrock** service
3. Go to **Model access** in the left sidebar
4. Click **Request model access** or **Manage model access**
5. Find **Llama 4 Maverick 17B Instruct** in the list
6. Click **Request access** and follow the prompts
7. Wait for approval (usually instant for most models)

**Note**: Model availability varies by AWS region. We recommend using **us-east-1** (US East - N. Virginia) as it typically has the most models available.

## AWS Credential Configuration

SANEval's Bedrock integration supports three authentication methods. Choose the one that best fits your environment.

### Method 1: Environment Variables (Recommended for CI/CD)

Set AWS credentials as environment variables. This is the simplest method for automated environments.

```bash
export AWS_ACCESS_KEY_ID=your_access_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_key_here
export AWS_DEFAULT_REGION=us-east-1
```

**To get your access keys:**
1. Log into AWS Console
2. Navigate to **IAM** (Identity and Access Management)
3. Go to **Users** → Select your user
4. Click **Security credentials** tab
5. Click **Create access key**
6. Save the Access Key ID and Secret Access Key securely

**Security Note**: Never commit access keys to version control. Use environment variables or AWS profiles.

### Method 2: AWS Profile (Recommended for Local Development)

Use AWS CLI profiles for local development. This keeps credentials separate from your code.

#### Step 1: Install AWS CLI

```bash
# macOS
brew install awscli

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Windows
# Download installer from https://aws.amazon.com/cli/
```

#### Step 2: Configure AWS Profile

```bash
aws configure --profile your_profile_name
```

You'll be prompted for:
- **AWS Access Key ID**: Your access key
- **AWS Secret Access Key**: Your secret key
- **Default region name**: `us-east-1` (recommended)
- **Default output format**: `json` (recommended)

#### Step 3: Set Profile in Environment

```bash
export AWS_PROFILE=your_profile_name
export AWS_DEFAULT_REGION=us-east-1
```

#### Step 4: Verify Configuration

```bash
aws sts get-caller-identity --profile your_profile_name
```

You should see output like:
```json
{
    "UserId": "AIDAXXXXXXXXXXXXXXXXX",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/your-username"
}
```

### Method 3: IAM Roles (Automatic in AWS Environments)

If running in AWS environments (ECS, Lambda, EC2), IAM roles are used automatically. No configuration needed!

**Supported environments:**
- Amazon ECS (Elastic Container Service)
- AWS Lambda
- Amazon EC2 instances with IAM roles
- AWS Fargate

**The integration automatically detects these environments and uses IAM role credentials.**

## Region Configuration

Llama 4 models are not available in all AWS regions. We recommend **us-east-1** for best model availability.

### Set Default Region

```bash
export AWS_DEFAULT_REGION=us-east-1
```

### Supported Regions (as of Dec 2024)

Check current availability at: https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html

Common regions with broad model support:
- `us-east-1` (US East - N. Virginia) ✅ Recommended
- `us-west-2` (US West - Oregon)
- `eu-west-1` (Europe - Ireland)
- `ap-southeast-1` (Asia Pacific - Singapore)

## Verifying Your Setup

### 1. Test AWS Credentials

```bash
aws sts get-caller-identity
```

### 2. Check Bedrock Model Access

```bash
aws bedrock list-foundation-models --region us-east-1
```

Look for Llama 4 models in the output:
```json
{
    "modelId": "us.meta.llama4-maverick-17b-instruct-v1:0",
    "modelName": "Llama 4 Maverick 17B Instruct"
}
```

### 3. Test with SANEval

```python
from ssa.vlm import Llm
from ssa.providers.bedrock import LLAMA_4_MAVERICK

# This should succeed if credentials are configured correctly
llm = Llm(LLAMA_4_MAVERICK)
response = llm.call("Hello, are you working?")
print(response)
```

## Common Issues and Troubleshooting

### Issue: "No AWS credentials found"

**Solution**: Ensure you've set up credentials using one of the three methods above.

```bash
# Check if environment variables are set
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY

# Or check if profile is set
echo $AWS_PROFILE
```

### Issue: "Profile not found"

**Solution**: Configure the AWS profile:

```bash
aws configure --profile your_profile_name
```

### Issue: "AccessDeniedException" or "UnauthorizedOperation"

**Possible causes:**
1. Your AWS credentials don't have Bedrock permissions
2. You haven't requested model access
3. Model not available in your region

**Solution**:

1. **Check IAM Permissions**: Ensure your IAM user/role has these permissions:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "bedrock:InvokeModel",
           "bedrock:ListFoundationModels"
         ],
         "Resource": "*"
       }
     ]
   }
   ```

2. **Request Model Access**: Follow the "Requesting Bedrock Model Access" section above

3. **Check Region**: Verify you're using a region where Llama 4 is available

### Issue: "Model not found" or "ValidationException"

**Solution**:
- Verify model access has been approved in AWS Bedrock console
- Check that you're using the correct region (us-east-1 recommended)
- Model IDs can change; verify the latest ID in AWS Bedrock console

### Issue: "Rate exceeded" or "ThrottlingException"

**Solution**:
- AWS Bedrock has rate limits and quotas
- Retry with exponential backoff (built into SANEval)
- Request quota increases in AWS Service Quotas console if needed

### Issue: "ExpiredToken" or "InvalidClientTokenId"

**Solution**:
- AWS credentials may have expired
- For profiles: Run `aws sso login --profile your_profile_name`
- For environment variables: Regenerate access keys in IAM console

## Security Best Practices

### 1. Never Commit Credentials

```bash
# Add to .gitignore
.env
.aws/
credentials
```

### 2. Use IAM Roles When Possible

IAM roles are more secure than access keys:
- No long-lived credentials to manage
- Automatically rotated
- Easier to audit

### 3. Principle of Least Privilege

Only grant the minimum required permissions:
```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel"
  ],
  "Resource": "arn:aws:bedrock:*:*:foundation-model/us.meta.llama4-*"
}
```

### 4. Rotate Access Keys Regularly

- Set up key rotation reminders
- Use AWS Secrets Manager for production
- Consider AWS IAM Identity Center (SSO) for organizations

### 5. Monitor Usage and Costs

- Enable AWS CloudTrail for audit logs
- Set up billing alerts in AWS Budgets
- Use SANEval's built-in cost tracking

## Cost Management

### Pricing Information

Llama 4 pricing on Bedrock (verify at https://aws.amazon.com/bedrock/pricing/):
- Input: ~$0.0008 per 1M tokens (estimated)
- Output: ~$0.001 per 1M tokens (estimated)

**Note**: Actual pricing may vary. Check AWS Bedrock pricing page for current rates.

### Track Costs in SANEval

```python
from ssa.utils.costs import reset_cost_tracking, get_total_cost

reset_cost_tracking()

# ... make your model calls ...

total = get_total_cost()
print(f"Total cost: ${total:.6f}")
```

### Set AWS Budget Alerts

1. Go to AWS Budgets in the console
2. Create a new budget
3. Set threshold alerts (e.g., $10, $50, $100)
4. Get notifications before overspending

## Next Steps

Once your AWS setup is complete:

1. **Run Examples**: `python examples/bedrock_llama_examples.py`
2. **Run Tests**: `pytest tests/unit/test_bedrock.py -v`
3. **Integration**: Start using Llama 4 in your SANEval workflows

## Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AWS CLI Configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)
- [SANEval GitHub Repository](https://github.com/yourusername/saneval)

## Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify AWS Bedrock service status: https://status.aws.amazon.com/
3. Review AWS CloudWatch logs for detailed error messages
4. Open an issue on the SANEval GitHub repository

---

Last updated: December 2024
