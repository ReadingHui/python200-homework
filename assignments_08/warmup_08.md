# Part 1: Warmup — Cloud Concepts
## Cloud Concepts Question 1
What is the core economic model of cloud computing, and how does it differ from owning your own servers?

### Answer:
The core economic model of cloud computing is the company building data-centers, and selling the computing and storage service to customers. It is different than owning my own server in that I don't have to pay for and maintain an actual server-side hardware, I can just rent the service provided by the providers and pay-as-I-go.

## Cloud Concepts Question 2
What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

* A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.
* A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.
* A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.

### Answer:
Vertical scaling is when we want more computing/storage power by upgrading the machine itself, e.g. more CPU, RAM, GPU. Horizontal scaling means adding more machines and splitting the work across them. For example, if I need to increase the complexity of my developed app, adding more functionality to it, I may need a vertical scaling to handle the increased burden per inference; on the other hand, if I am expanded my service to more countries, with an expected surge of users, I will choose horizontal scaling and buy/rent more machines to coup with the increased simultaneous requests.

Cases:
* Horizontal scaling, as more simultaneous usage, but not increase in complexity.
* Vertical scaling, job too complex for the current set-up, need to increase GPU and RAM
* Horizontal scaling, increment in parallel processing.

## Cloud Concepts Question 3
Before writing your definitions, classify each item in the list below as IaaS, PaaS, SaaS, or BaaS. One sentence of reasoning is enough for each.

* Gmail
* Azure Virtual Machines
* AWS S3 (Simple Storage Service)
* GitHub Codespaces
* Snowflake
* Supabase

Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.

### Answer:
* Gmail: SaaS, it is a serverless compute running on the Google App Engine
* Azure Virtual Machines: IaaS, it provides customization down to hardware and OS
* AWS S3: IaaS, it provides object storage
* GitHub Codespaces: PaaS, it provides a compute platform of sandbox web application deployment space
* Snowflake: SaaS, it is a cloud data platform for web-based data warehouse and data lake
* Supabase: BaaS, it provides a cloud backend for PostgreSQL database access

#### IaaS:
Infrastructure-as-a-Service means except the physical hardware, the user basically has control over all other aspect, from virtual machine OS, environment, software to updates.

Example: Azure Virtual Machines

As a developer, I would be responsible for 
- OS installation, configuration and maintenance
- Network configuration
- Database and storage configuration

#### PaaS:
Platform-as-a-Service means the infrastructure of the cloud service is maintained by the provider, as a user we use the service as a platform to deploy and maintain our application and data.

Example: Google App Engine

As a developer, I would be responsible for 
- Deployment and maintenance of applications
- Data
- Access control

#### SaaS:
Software-as-a-Service means I am using/renting the developed software directly, so I don't have to develop my own.

Example: Gmail

As a developer, I would be responsible for 
- Data
- Access control

## Cloud Concepts Question 4
What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like AWS or GCP directly? What do you gain, and what do you give up?

### Answer:
A managed data platform is an SaaS service, where all the data analytic tools such as dashboard are already provided, we only need to upload or build a data pipeline to put the data into those platform, then we can do all sorts of data analysis/engineering on the platform. Compare to using a cloud provider directly, we don't need to set up OS, build/install software or do the coding by ourselves, we just need to pour the data in, all the tools and codes are built for us to use already. We gain the convenience and speed in deployment, but we lose out on the customization and control on the base software/tools.

## Cloud Concepts Question 5
The lesson names two situations where the cloud is probably not the right choice. What are they?

### Answer:
- Dataset fits comfortably on a single machine and you do not have massive compute demands
- Cost is too steep for a simple application

# Part 2: Warmup — Cloud Landscape
## Cloud Landscape Question 1
Name the three hyperscalers. For each, write one sentence describing its primary strength and the type of organization most likely to use it.
### Answer:
* Amazon Web Services (AWS): It has the broadest service catalog, large enterprise, startup or nonprofit with engineering staffs are most likely to use it.
* Google Cloud Platform (GCP): It is the strongest in data and machine learning, AI startups and tech business that wants to train AI models are most likely to use it.
* Microsoft Azure: It is strongest in deep integration with Windows, Active Directory, and Microsoft 365. Most likely users are large nonporfits and public-sector organizations, and governments.

## Cloud Landscape Question 2

The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words — one sentence each.

Then add your own reflection: what does this suggest about how you should evaluate a cloud tool when starting a new project?

### Answer:
#### Reasons to switch to Supabase:
* Access - getting in Azure need to wait for verification and configuring authentication, which could be problematic for students who join later, or just running into configuration problem walls in general.
* Pedagogical fit - Azure stores files as files and organized by path, while Supabase stores them as rows and columns in relational database, which we can query, filter etc. much easier when our goal is to manipulate the data.
* Pipeline coherence - the pipeline we are building in the future will be using two zones, which nicely fits into 2 tables in Supabase.

#### Reflection:
When starting a new project, it is important to evaluate the scale and the service that I would need to use, as sometimes the larger provider may not provide the best fit service, like the data storage in Azure does not match with what our course is designed to be, Supabase is actually a better fit.

## Cloud Landscape Question 3

For each of the four scenarios below, identify which service category from the taxonomy table applies (e.g., "object storage", "managed relational DB", "LLM API", "serverless compute") and name one specific provider or product that offers it.

1. You need to store 10 TB of image files and retrieve them by filename from any machine.
2. You need to run an ML training job on a GPU for four hours, then shut it down.
3. You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets.
4. You need to send structured data to a large language model and get a text response back.

### Answer:
1. Object storage, AWS S3 does it.
2. ML Platform, Azure ML does it.
3. Serverless compute, GCP Cloud Functions does it.
4. LLM API, Vertex AI does it.

## Cloud Landscape Question 4

The lesson says most projects don't use one provider for everything. Describe a simple data project of your own design (one or two sentences is fine) and sketch a plausible stack using services from at least two different providers or products from the taxonomy table. Then answer: is there a benefit to consolidating to one provider, and what would you give up if you did?

### Answer:
A simple image classification project of Pokemon Cards, where I could be using Supabase for the relational database to store the map between the name/id of the card and the details of the card (like price, type, foil treatments), Vertex AI for the actual model training and deployment, and AWS S3 for the image storage for comparison.

Consolidating to one provider limits the possibility of one service being down affecting the whole project, and easier integration between the moving parts. However, I will be giving up on the flexibility of choosing the best service out there, and the easy access to some like Supabase.

# Introduction video
[Video Link](https://youtu.be/YBZXA_C5ubE)