# Deployment Guide

This guide covers different deployment options for the Reddit Sentiment Dashboard.

---

## 🐳 Docker Deployment (Recommended)

### Prerequisites
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose (included with Docker Desktop)

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/DharmpratapSingh/Reddit-Sentiment-Dashboard.git
cd Reddit-Sentiment-Dashboard
```

2. **Create .env file**:
```bash
cp .env.example .env
# Edit .env with your Reddit API credentials
```

3. **Build and run with Docker Compose**:
```bash
docker-compose up -d
```

4. **Access the application**:
Open http://localhost:8501 in your browser

### Docker Commands

```bash
# Build the image
docker build -t reddit-sentiment-dashboard .

# Run the container
docker run -d -p 8501:8501 --env-file .env reddit-sentiment-dashboard

# View logs
docker logs reddit-sentiment-dashboard

# Stop the container
docker stop reddit-sentiment-dashboard

# Remove the container
docker rm reddit-sentiment-dashboard
```

---

## ☁️ Cloud Deployment

### 1. Streamlit Cloud (Easiest)

**Pros**: Free, automatic HTTPS, built-in secrets management
**Cons**: Limited resources, public repos only (or pay)

**Steps**:

1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Click "New app"
4. Select your repository and branch
5. Set `main_code.py` as the main file
6. Add secrets in Settings → Secrets:
```toml
REDDIT_CLIENT_ID = "your_client_id"
REDDIT_CLIENT_SECRET = "your_secret"
REDDIT_USER_AGENT = "your_user_agent"
```
7. Click Deploy

**Cost**: Free (with limits)

---

### 2. Heroku

**Pros**: Easy deployment, scalable, add-ons available
**Cons**: Paid plans required for better performance

**Steps**:

1. **Install Heroku CLI**:
```bash
curl https://cli-assets.heroku.com/install.sh | sh
```

2. **Create Heroku app**:
```bash
heroku create reddit-sentiment-dashboard
```

3. **Create `Procfile`**:
```
web: streamlit run main_code.py --server.port=$PORT --server.address=0.0.0.0
```

4. **Set environment variables**:
```bash
heroku config:set REDDIT_CLIENT_ID="your_id"
heroku config:set REDDIT_CLIENT_SECRET="your_secret"
heroku config:set REDDIT_USER_AGENT="your_agent"
```

5. **Deploy**:
```bash
git push heroku main
```

6. **Open app**:
```bash
heroku open
```

**Cost**: $7-$50/month depending on dyno type

---

### 3. AWS (Elastic Beanstalk)

**Pros**: Highly scalable, AWS ecosystem integration
**Cons**: More complex setup, can be expensive

**Steps**:

1. **Install EB CLI**:
```bash
pip install awsebcli
```

2. **Initialize EB**:
```bash
eb init -p docker reddit-sentiment-dashboard
```

3. **Create environment**:
```bash
eb create reddit-sentiment-env
```

4. **Set environment variables**:
```bash
eb setenv REDDIT_CLIENT_ID=your_id REDDIT_CLIENT_SECRET=your_secret
```

5. **Deploy**:
```bash
eb deploy
```

6. **Open app**:
```bash
eb open
```

**Cost**: ~$15-100/month depending on instance type

---

### 4. Google Cloud Run

**Pros**: Serverless, pay-per-use, auto-scaling
**Cons**: Cold starts, request timeout limits

**Steps**:

1. **Install gcloud CLI**: https://cloud.google.com/sdk/docs/install

2. **Authenticate**:
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

3. **Build and push Docker image**:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/reddit-sentiment
```

4. **Deploy to Cloud Run**:
```bash
gcloud run deploy reddit-sentiment \
  --image gcr.io/YOUR_PROJECT_ID/reddit-sentiment \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars REDDIT_CLIENT_ID=your_id,REDDIT_CLIENT_SECRET=your_secret
```

5. **Access the service**:
The deployment will output the service URL

**Cost**: ~$5-50/month depending on usage

---

### 5. DigitalOcean App Platform

**Pros**: Simple deployment, predictable pricing
**Cons**: Limited to DigitalOcean ecosystem

**Steps**:

1. Go to https://cloud.digitalocean.com/apps
2. Click "Create App"
3. Connect your GitHub repository
4. Select branch and configure:
   - **Source Directory**: `/`
   - **Environment**: Docker
   - **HTTP Port**: 8501
5. Add environment variables
6. Click "Deploy"

**Cost**: $12-25/month

---

### 6. Azure Container Instances

**Pros**: Simple container deployment, Azure integration
**Cons**: Not as feature-rich as other Azure services

**Steps**:

1. **Install Azure CLI**: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

2. **Login**:
```bash
az login
```

3. **Create resource group**:
```bash
az group create --name reddit-sentiment-rg --location eastus
```

4. **Create container registry**:
```bash
az acr create --resource-group reddit-sentiment-rg --name redditsentimentacr --sku Basic
```

5. **Build and push image**:
```bash
az acr build --registry redditsentimentacr --image reddit-sentiment:latest .
```

6. **Deploy container**:
```bash
az container create \
  --resource-group reddit-sentiment-rg \
  --name reddit-sentiment \
  --image redditsentimentacr.azurecr.io/reddit-sentiment:latest \
  --dns-name-label reddit-sentiment-unique \
  --ports 8501 \
  --environment-variables REDDIT_CLIENT_ID=your_id REDDIT_CLIENT_SECRET=your_secret
```

**Cost**: ~$10-30/month

---

## 🖥️ VPS Deployment (Self-Hosted)

### Prerequisites
- VPS with Ubuntu 20.04+ (e.g., DigitalOcean Droplet, AWS EC2, Linode)
- Domain name (optional but recommended)
- SSH access to the server

### Steps

1. **SSH into your server**:
```bash
ssh root@your-server-ip
```

2. **Install Docker**:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

3. **Install Docker Compose**:
```bash
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```

4. **Clone repository**:
```bash
git clone https://github.com/DharmpratapSingh/Reddit-Sentiment-Dashboard.git
cd Reddit-Sentiment-Dashboard
```

5. **Create .env file**:
```bash
nano .env
# Add your credentials
```

6. **Start the application**:
```bash
docker-compose up -d
```

7. **Set up Nginx reverse proxy** (optional but recommended):
```bash
apt install nginx
nano /etc/nginx/sites-available/reddit-sentiment
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
ln -s /etc/nginx/sites-available/reddit-sentiment /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx
```

8. **Set up SSL with Let's Encrypt**:
```bash
apt install certbot python3-certbot-nginx
certbot --nginx -d your-domain.com
```

9. **Set up auto-restart**:
```bash
# Add to crontab
crontab -e
```

```
@reboot cd /root/Reddit-Sentiment-Dashboard && docker-compose up -d
```

---

## 🔄 Continuous Deployment

### GitHub Actions Auto-Deploy

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Deploy to Server
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.SERVER_HOST }}
        username: ${{ secrets.SERVER_USER }}
        key: ${{ secrets.SSH_PRIVATE_KEY }}
        script: |
          cd /root/Reddit-Sentiment-Dashboard
          git pull origin main
          docker-compose down
          docker-compose up -d --build
```

Add these secrets to your GitHub repository:
- `SERVER_HOST`: Your server IP
- `SERVER_USER`: SSH username
- `SSH_PRIVATE_KEY`: Your SSH private key

---

## 📊 Monitoring & Maintenance

### Health Checks

```bash
# Check if container is running
docker ps | grep reddit-sentiment

# Check logs
docker logs -f reddit-sentiment-dashboard

# Check resource usage
docker stats reddit-sentiment-dashboard
```

### Automated Backups

```bash
# Create backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker exec reddit-sentiment-dashboard tar czf /tmp/backup_$DATE.tar.gz /app/data
docker cp reddit-sentiment-dashboard:/tmp/backup_$DATE.tar.gz ./backups/
```

### Update Strategy

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up -d --build

# Or without downtime
docker-compose up -d --build --no-deps reddit-sentiment-dashboard
```

---

## 🔐 Security Best Practices

1. **Never commit .env files**
2. **Use HTTPS in production**
3. **Set up firewall rules**:
```bash
ufw allow 22    # SSH
ufw allow 80    # HTTP
ufw allow 443   # HTTPS
ufw enable
```
4. **Keep Docker images updated**
5. **Use secrets management** (AWS Secrets Manager, Azure Key Vault, etc.)
6. **Enable rate limiting** at the reverse proxy level
7. **Set up monitoring** (Prometheus, Grafana, or cloud provider tools)

---

## 🆘 Troubleshooting

### Container won't start

```bash
# Check logs
docker logs reddit-sentiment-dashboard

# Check if port is already in use
netstat -tulpn | grep 8501

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### Out of Memory

```bash
# Check memory usage
docker stats

# Increase container memory limit in docker-compose.yml
services:
  reddit-sentiment-dashboard:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Slow Performance

- Reduce number of posts analyzed
- Use model caching
- Increase server resources
- Enable Redis for caching (advanced)

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)
- [NGINX Configuration](https://nginx.org/en/docs/)
- [Let's Encrypt](https://letsencrypt.org/)

---

## 💬 Need Help?

Open an issue on GitHub: https://github.com/DharmpratapSingh/Reddit-Sentiment-Dashboard/issues
