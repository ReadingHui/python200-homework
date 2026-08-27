# Part A: Supabase Setup
Project is set up.

It seems like Supabase updated how the API keys are handled, the `anon` `public` key is moved to a `legacy` page, although still usable.

# Part B: Cloud Cost Analysis

## Scenario A — Lightweight compute: 
* t3.micro EC2 instance (1 vCPU, 1 GB RAM)
* On-demand pricing
* Running 8 hours per day, 5 days per week (approximately 160 hours per month). 
* Use the US East (N. Virginia) region.

### Cost
- Total 12 months cost - 19.92 USD - Includes upfront cost

## Scenario B — Heavy analytics workload: 
* p3.2xlarge EC2 instance (8 vCPU, 1 V100 GPU) 
* Running 24/7 for the full month (730 hours)
* An RDS db.m5.large instance (2 vCPU, 8 GB RAM)
* S3 Standard storage bucket with 1 TB of data. 
* Use US East (N. Virginia).

### Cost
- Total 12 months cost - 30,955.80 USD - Includes upfront cost

## Summary
The prices do surprise me that it is not actually quite cheap tp set up a basic EC2 instance, but it could scale up really quickly as the demand increases. It is really interesting to see how customizable a cloud service can be, with all the options, there are thousand to millions of combinations we can get, which in itself could be a headache too. The second instance is so much more expensive, and the cost mainly came from the EC2 instance with the GPU attached. Hence, unless we are doing some heavy GPU-relying tasks like model training, it should be generally not worth to go for them.
