# Deployment

Three shapes are provided. They differ in cost and in what they can guarantee.

| | Where | Cost | Multi-replica | Notes |
|---|---|---|---|---|
| **Local** | [`docker-compose.yml`](../docker-compose.yml) | — | no | SQLite, one API container |
| **Scale-out** | [`docker-compose.scale.yml`](../docker-compose.scale.yml) | — | yes | PostgreSQL, N replicas behind nginx |
| **Kubernetes** | [`kubernetes/`](kubernetes/) | varies | yes | HPA, PDB, headless Service for promotion reloads |
| **AWS single node** | [`terraform/`](terraform/) | ~$18–20/mo | no | **Currently deployed** |

Measured capacity and what does not scale: [`docs/scaling.md`](../docs/scaling.md).

---

## AWS single node

One `t3.small` runs the API, the outbox worker and PostgreSQL under docker
compose, behind an Elastic IP. It shares nothing with other stacks in the
account.

### What is running

| Component | Notes |
|---|---|
| `api` | Published on port 80. One replica; ~32 rps ceiling per worker process. |
| `worker` | Drains the transactional outbox. |
| `postgres` | Container on the same host, data on the encrypted root volume. |
| `scheduler` | **Not running** — see below. |

### Why the scheduler is not running

It retrains, which needs the training dataset — and that dataset contains
customer geography at household granularity. Putting it in this account is a data
movement decision, not a deployment detail, so serving runs and the lifecycle does
not. To enable it: upload `data/Telco_customer_churn_raw.csv` to
`s3://<artifacts-bucket>/data/`, extend the bootstrap to sync it, and start the
`scheduler` service.

### Access

There is no SSH key pair and no inbound port 22. Open a shell with Session
Manager, which authenticates via IAM and is audited in CloudTrail:

```bash
aws ssm start-session --target <instance-id> --profile aiops-deploy --region us-east-1
```

Container logs go to the `/churn-ml/prod` CloudWatch log group. The bootstrap's
own log is at `/var/log/churn-bootstrap.log` on the instance — the first place to
look if the API is not answering.

### Secrets

Nothing secret is baked into the AMI, the user-data or the repository. The
instance reads five SecureString parameters at boot using its instance role:

```
/churn-ml/prod/api_key
/churn-ml/prod/admin_api_key
/churn-ml/prod/artifact_signing_key
/churn-ml/prod/subject_key_salt
/churn-ml/prod/postgres_password
```

Retrieve the API key clients must send as `X-API-Key`:

```bash
aws ssm get-parameter --name /churn-ml/prod/api_key --with-decryption \
  --query Parameter.Value --output text --profile aiops-deploy --region us-east-1
```

### The model bundle

The signed bundle lives at `s3://<artifacts-bucket>/model/production/current/`.
It is signed with the **production** signing key, and the serving layer verifies
that signature *before* unpickling `model.pkl` — so a tampered object in S3 fails
to load rather than executing.

The instance role can read the `model/` prefix and nothing else in the bucket.

### Deploying a new version

```bash
# 1. Build and push, tagged with the commit being deployed.
TAG=$(git rev-parse --short HEAD)
REG=<account>.dkr.ecr.us-east-1.amazonaws.com
aws ecr get-login-password --profile aiops-deploy | docker login --username AWS --password-stdin $REG
docker build -f Dockerfile          -t $REG/churn-ml-api:$TAG .
docker build -f Dockerfile.training -t $REG/churn-ml-training:$TAG .
docker push $REG/churn-ml-api:$TAG
docker push $REG/churn-ml-training:$TAG

# 2. Roll the instance onto it. user_data_replace_on_change means changing the
#    tag replaces the instance, so the new one boots on exactly this code.
cd deploy/terraform
terraform apply -var="image_tag=$TAG"
```

The tag is pinned to a commit rather than `latest` on purpose: a replacement
instance must come up running the same code as its predecessor, and `latest`
breaks that the moment anyone pushes.

Promoting a **model** is separate from deploying **code** — re-sign the bundle
with the production key, sync it to S3, and either restart the API or call
`POST /admin/reload-model`.

### Teardown

```bash
cd deploy/terraform && terraform destroy
```

The S3 buckets and SSM parameters are managed outside Terraform (they hold state
and secrets that must survive a destroy) and need removing by hand if you want
them gone.

### Verifying a deploy

```bash
KEY=$(aws ssm get-parameter --name /churn-ml/prod/api_key --with-decryption \
        --query Parameter.Value --output text --profile aiops-deploy --region us-east-1)
bash scripts/smoke_deployment.sh http://<public-ip> "$KEY"
```

Nine checks, covering the things that actually break: `/ready` runs a real
prediction, authentication is enforced, old and new clients both work **and
agree**, a misspelled field is still a 422, and the served threshold comes from
the bundle rather than a hardcoded 0.5.

The old-versus-new agreement check is the important one. If those two answers ever
differ, the geographic fields are influencing the model again — which is the exact
thing removing them was meant to stop.

### Known limits

* **Single node.** No rolling deploys, and a replacement instance is a brief
  outage. The multi-replica work — PostgreSQL advisory-lock leader election, a
  shared bundle volume, HPA — is in [`kubernetes/`](kubernetes/) for when this
  outgrows one box.
* **HTTP only.** No certificate, so the API key travels in clear text. Put the
  instance behind an ALB with an ACM certificate, or a CloudFront distribution,
  before treating this as anything but a demo.
* **PostgreSQL is on the instance volume.** It is encrypted and snapshot-capable
  but has no automated backup; losing the instance loses the prediction history.
