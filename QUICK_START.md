# 🚀 Quick Start Guide - Trading Bot Deployment

## ปัญหา: GitHub Actions Error 451
Binance API ถูก block บน GitHub Actions (US servers)

---

## ✅ แนะนำ 3 วิธีที่ดีที่สุด

### 1️⃣ PythonAnywhere (ง่ายสุด - แนะนำสำหรับ Beginner)

**Setup 5 นาที:**

1. ไป https://www.pythonanywhere.com/ → Sign up (ฟรี)
2. เปิด Bash console
3. รัน:
   ```bash
   git clone https://github.com/YOUR_USERNAME/trading-bot.git
   cd trading-bot
   chmod +x pythonanywhere_setup.sh
   ./pythonanywhere_setup.sh
   ```
4. แก้ `.env` file:
   ```bash
   nano .env
   # ใส่ email และ API keys
   ```
5. ไปที่ **Tasks** tab → Add scheduled task:
   ```
   Command: cd ~/trading-bot && source venv/bin/activate && python main.py deepseek
   Schedule: Daily at 23:30 UTC
   ```

✅ เสร็จแล้ว!

---

### 2️⃣ Railway.app (แนะนำสำหรับ Production)

**Setup 3 นาที:**

1. ไป https://railway.app/ → Login with GitHub
2. New Project → Deploy from GitHub repo
3. เลือก `trading-bot` repo
4. Add Variables:
   ```
   BOT_EMAIL_SENDER=your@gmail.com
   BOT_EMAIL_PASSWORD=app-password
   BOT_EMAIL_RECIPIENT=recipient@email.com
   DEEPSEEK_API_KEY=sk-xxx
   ```
5. Settings → Cron:
   - Schedule: `30 23 * * *`
   - Command: `python main.py deepseek`
6. Deploy!

✅ Auto-deploy ทุกครั้งที่ push!

---

### 3️⃣ Render.com (ฟรี 750 ชม./เดือน)

**Setup 4 นาที:**

1. ไป https://render.com/ → Sign up
2. New → Cron Job
3. Connect repository: `trading-bot`
4. Render จะอ่าน `render.yaml` อัตโนมัติ
5. เพิ่ม Environment Variables (เหมือน Railway)
6. Create Cron Job

✅ เสร็จแล้ว!

---

## 🐳 Docker (สำหรับคนใช้ VPS/Server)

**รันครั้งเดียว:**
```bash
docker-compose run --rm trading-bot
```

**รันเป็น scheduled job:**
```bash
docker-compose up -d
```

---

## 📊 เปรียบเทียบ

| แพลตฟอร์ม | ฟรี? | ง่าย | Auto-deploy | Cron |
|-----------|------|------|-------------|------|
| **PythonAnywhere** | ✅ | ⭐⭐⭐⭐⭐ | ❌ Manual | ✅ |
| **Railway** | $5/mo | ⭐⭐⭐⭐ | ✅ | ✅ |
| **Render** | 750h/mo | ⭐⭐⭐⭐ | ✅ | ✅ |
| **Docker** | ฟรี (ถ้ามี VPS) | ⭐⭐⭐ | ❌ | ✅ |

---

## ❓ คำถามที่พบบ่อย

**Q: ทำไมต้องเปลี่ยนจาก GitHub Actions?**
A: Binance API block GitHub servers (Error 451)

**Q: แพลตฟอร์มไหนดีสุด?**
A:
- **Beginner**: PythonAnywhere
- **Developer**: Railway
- **Budget**: Render (750 ชม. ฟรี)

**Q: ต้องเสียเงินไหม?**
A: ไม่! ทุกแพลตฟอร์มมี free tier พอใช้งาน daily bot

**Q: ถ้า deploy แล้วยังไม่ทำงาน?**
A: Check logs และดู Environment Variables ว่าถูกต้องไหม

---

## 📝 หลังจาก Deploy

1. ตรวจสอบ logs ว่า bot รันได้
2. รอรับอีเมลตอน 7-8 โมงเช้า
3. ถ้ามีปัญหา ดู logs ใน dashboard

---

## 🆘 ต้องการความช่วยเหลือ?

เลือก platform ที่ชอบแล้วบอก จะช่วย setup step-by-step! 🚀

---

**Created by:** Trading Bot Enhanced with DeepSeek AI
**Last Updated:** 2025-11-25
