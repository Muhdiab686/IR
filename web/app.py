# web/app.py

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

# إنشاء تطبيق FastAPI جديد للواجهة
app = FastAPI()

# تحديد مسار المجلدات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# 1. خدمة الملفات الثابتة (CSS, JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 2. إعداد قوالب Jinja2
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# 3. إنشاء Endpoint رئيسي لعرض صفحة البحث
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    هذا الـ Endpoint يعرض صفحة البحث الرئيسية index.html
    """
    return templates.TemplateResponse("index.html", {"request": request})