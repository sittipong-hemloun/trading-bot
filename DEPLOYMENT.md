# Trading Bot Deployment Guide

## ปัญหา GitHub Actions
GitHub Actions มี HTTP 451 Error เพราะ Binance API ถูก block ในบางภูมิภาค (US servers)

## แนะนำ: Railway.app (ที่ดีที่สุด)

### ทำไมต้อง Railway?
- ✅ ฟรี $5 credit/เดือน (พอสำหรับ daily cron)
- ✅ ไม่ block Binance API
- ✅ รองรับ scheduled jobs
- ✅ Deploy ง่าย จาก GitHub
- ✅ มี logs ดูได้

### วิธี Deploy บน Railway:

1. **สร้างบัญชี Railway**
   ```
   https://railway.app/
   ```
   Login ด้วย GitHub

2. **New Project → Deploy from GitHub repo**
   - เลือก repository `trading-bot`
   - Branch: `main`

3. **Add Environment Variables**
   ```
   BOT_EMAIL_SENDER=your@gmail.com
   BOT_EMAIL_PASSWORD=your-app-password
   BOT_EMAIL_RECIPIENT=recipient@email.com
   DEEPSEEK_API_KEY=your-deepseek-key
   ```

4. **เพิ่ม Cron Job**
   - ไปที่ Settings → Cron
   - Schedule: `30 23 * * *`
   - Command: `python main.py deepseek`

5. **Deploy!**
   - Railway จะ auto-deploy ทุกครั้งที่ push

---

## ทางเลือกอื่น

### Option 2: Render.com

**Setup:**
1. Fork repo
2. สร้าง Cron Job ใน Render
3. เลือก `render.yaml` (มีใน repo แล้ว)
4. เพิ่ม environment variables
5. Deploy

**Free Tier:**
- 750 ชม./เดือน
- รัน cron job ฟรี

**Link:** https://render.com/

---

### Option 3: PythonAnywhere (ง่ายสุด)

**Setup:**
1. สร้างบัญชีฟรี: https://www.pythonanywhere.com/
2. Upload ไฟล์ทั้งหมด
3. ตั้ง scheduled task:
   ```bash
   cd ~/trading-bot && python3 main.py deepseek
   ```
4. Schedule: Daily at 23:30 UTC

**Free Tier:**
- 1 scheduled task/day
- เพียงพอสำหรับ bot นี้

---

### Option 4: Fly.io

**Setup:**
```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Deploy
flyctl launch

# Set secrets
flyctl secrets set BOT_EMAIL_SENDER=xxx
flyctl secrets set BOT_EMAIL_PASSWORD=xxx
flyctl secrets set BOT_EMAIL_RECIPIENT=xxx
flyctl secrets set DEEPSEEK_API_KEY=xxx

# Deploy
flyctl deploy
```

**Free Tier:**
- 3 shared-cpu-1x VMs
- 160GB outbound data transfer

---

## เปรียบเทียบ

| Platform | Free Tier | Cron Support | Binance API | Ease |
|----------|-----------|--------------|-------------|------|
| **Railway** | $5/mo credit | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| **Render** | 750 hrs | ✅ | ✅ | ⭐⭐⭐⭐ |
| **PythonAnywhere** | 1 task/day | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| **Fly.io** | 3 VMs | ⚠️ (manual) | ✅ | ⭐⭐⭐ |
| GitHub Actions | Unlimited | ✅ | ❌ | ⭐⭐⭐⭐⭐ |

---

## คำแนะนำ

**สำหรับ Beginner:** PythonAnywhere (ง่ายที่สุด, ไม่ต้อง config อะไรมาก)

**สำหรับ Production:** Railway (stable, ดี, มี logs ครบ)

**สำหรับคนชอบ CLI:** Fly.io (มี control เยอะ)

---

## หมายเหตุ

- ทุก platform สามารถเชื่อมต่อ Binance API ได้ (ไม่มี 451 error)
- Railway และ Render auto-deploy เมื่อ push to GitHub
- PythonAnywhere ต้อง upload manual หรือใช้ git pull

---

## ต้องการความช่วยเหลือ?

เลือก platform ที่ชอบแล้วบอก จะช่วย setup step-by-step! 🚀
