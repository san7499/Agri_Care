import os
import math
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin,
    login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ------------------ APP CONFIG ------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///agricare.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ------------------ LOGIN MANAGER ------------------
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ------------------ MODELS ------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # user / shop
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)


class Shop(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    shop_name = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    address = db.Column(db.String(300), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)


class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    shop_id = db.Column(db.Integer, db.ForeignKey("shop.id"), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    category = db.Column(db.String(100))
    price = db.Column(db.Float)
    description = db.Column(db.String(300))


class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    filename = db.Column(db.String(200))
    prediction = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    fertilizer = db.Column(db.String(500))
    pesticide = db.Column(db.String(500))
    organic = db.Column(db.String(500))


# ------------------ USER LOADER ------------------
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ------------------ ML MODEL ------------------
model = load_model("best_effb3_model.keras")

import json

with open("class_indices.json") as f:
    class_indices = json.load(f)

# reverse mapping
class_labels = {v: k for k, v in class_indices.items()}


# ------------------ TREATMENT ------------------
treatment_suggestions = {

    # 🌽 CORN
    "Corn__Blight": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Mancozeb fungicide"],
        "Organic": ["Crop rotation", "Remove infected leaves"]
    },
    "Corn__Common_Rust": {
        "Fertilizer": ["Nitrogen-rich fertilizer"],
        "Pesticide": ["Propiconazole"],
        "Organic": ["Neem oil spray"]
    },
    "Corn__Gray_Leaf_Spot": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Chlorothalonil"],
        "Organic": ["Proper field sanitation"]
    },
    "Corn__Healthy": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Not required"],
        "Organic": ["Organic compost"]
    },

    # 🍇 GRAPE
    "Grape___Black_rot": {
        "Fertilizer": ["Phosphorus-rich fertilizer"],
        "Pesticide": ["Mancozeb"],
        "Organic": ["Remove infected fruits"]
    },
    "Grape___Esca": {
        "Fertilizer": ["Organic manure"],
        "Pesticide": ["Thiophanate-methyl"],
        "Organic": ["Prune infected wood"]
    },
    "Grape___Leaf_blight": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Copper fungicide"],
        "Organic": ["Improve air circulation"]
    },
    "Grape___healthy": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Not required"],
        "Organic": ["Mulching"]
    },

    # 🥭 MANGO
    "Mango__Gall_Midge": {
        "Fertilizer": ["Organic compost"],
        "Pesticide": ["Imidacloprid"],
        "Organic": ["Sticky traps"]
    },
    "Mango__Healthy": {
        "Fertilizer": ["Farmyard manure"],
        "Pesticide": ["Not required"],
        "Organic": ["Regular pruning"]
    },
    "Mango__Powdery_Mildew": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Sulfur fungicide"],
        "Organic": ["Neem oil spray"]
    },
    "Mango__Sooty_Mould": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Dimethoate"],
        "Organic": ["Control aphids"]
    },

    # 🥜 PEANUT
    "Peanut__early_leaf_spot": {
        "Fertilizer": ["Calcium-rich fertilizer"],
        "Pesticide": ["Chlorothalonil"],
        "Organic": ["Crop rotation"]
    },
    "Peanut__early_rust": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Mancozeb"],
        "Organic": ["Remove infected plants"]
    },
    "Peanut__healthy_leaf": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Not required"],
        "Organic": ["Organic compost"]
    },
    "Peanut__late_leaf_spot": {
        "Fertilizer": ["Phosphorus fertilizer"],
        "Pesticide": ["Tebuconazole"],
        "Organic": ["Proper spacing"]
    },
    "Peanut__nutrition_deficiency": {
        "Fertilizer": ["Micronutrient mixture"],
        "Pesticide": ["Not required"],
        "Organic": ["Soil testing"]
    },
    "Peanut__rust": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Hexaconazole"],
        "Organic": ["Neem oil spray"]
    },

    # 🌶️ PEPPER
    "Pepper__bell___Bacterial_spot": {
        "Fertilizer": ["Calcium nitrate"],
        "Pesticide": ["Copper fungicide"],
        "Organic": ["Remove infected leaves"]
    },
    "Pepper__bell___healthy": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Not required"],
        "Organic": ["Compost manure"]
    },

    # 🥔 POTATO
    "Potato___Early_blight": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Chlorothalonil"],
        "Organic": ["Neem oil spray"]
    },
    "Potato___Late_blight": {
        "Fertilizer": ["Phosphorus fertilizer"],
        "Pesticide": ["Metalaxyl"],
        "Organic": ["Remove infected plants"]
    },
    "Potato___healthy": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Not required"],
        "Organic": ["Crop rotation"]
    },

    # 🍅 TOMATO
    "Tomato_Bacterial_spot": {
        "Fertilizer": ["Calcium-rich fertilizer"],
        "Pesticide": ["Copper fungicide"],
        "Organic": ["Neem oil spray"]
    },
    "Tomato_Early_blight": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Chlorothalonil"],
        "Organic": ["Remove infected leaves"]
    },
    "Tomato_Late_blight": {
        "Fertilizer": ["Phosphorus fertilizer"],
        "Pesticide": ["Metalaxyl"],
        "Organic": ["Crop rotation"]
    },
    "Tomato_Leaf_Mold": {
        "Fertilizer": ["Nitrogen fertilizer"],
        "Pesticide": ["Mancozeb"],
        "Organic": ["Improve ventilation"]
    },
    "Tomato_Septoria_leaf_spot": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Chlorothalonil"],
        "Organic": ["Remove infected leaves"]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "Fertilizer": ["Nitrogen fertilizer"],
        "Pesticide": ["Abamectin"],
        "Organic": ["Neem oil spray"]
    },
    "Tomato__Target_Spot": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Mancozeb"],
        "Organic": ["Crop sanitation"]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Imidacloprid (for whiteflies)"],
        "Organic": ["Yellow sticky traps"]
    },
    "Tomato__Tomato_mosaic_virus": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Not effective"],
        "Organic": ["Remove infected plants"]
    },
    "Tomato_healthy": {
        "Fertilizer": ["Balanced NPK fertilizer (20:20:20)"],
        "Pesticide": ["Not required"],
        "Organic": ["Organic compost"]
    }
}
# ------------------ HELPERS ------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image, ImageOps

def predict_disease(img_path):
    # Load image safely
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)   # fixes mobile rotation
    img = img.convert("RGB")
    img = img.resize((300, 300))

    # Convert to array
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ SAME preprocessing as training
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(preds))

    label = class_labels[idx]
    confidence = round(float(preds[idx]) * 100, 2)

    treatment = treatment_suggestions.get(label, {
        "Fertilizer": ["Consult agriculture expert"],
        "Pesticide": ["Consult agriculture expert"],
        "Organic": ["Consult agriculture expert"]
    })

    return label, confidence, treatment


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(d_lon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ------------------ AUTH ------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        if User.query.filter_by(username=request.form["username"]).first():
            flash("Username already exists")
            return redirect(url_for("signup"))

        user = User(
            username=request.form["username"],
            password=generate_password_hash(request.form["password"]),
            role=request.form["role"]
        )
        db.session.add(user)
        db.session.commit()
        flash("Account created")
        return redirect(url_for("login"))

    return render_template("signup.html")


# ------------------ LOGIN ------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()

        if user and check_password_hash(user.password, request.form["password"]):

            lat = request.form.get("latitude")
            lon = request.form.get("longitude")

            try:
                user.latitude = float(lat) if lat else None
                user.longitude = float(lon) if lon else None
            except ValueError:
                user.latitude = None
                user.longitude = None

            db.session.commit()

            login_user(user)

            return redirect(
                url_for("shop_dashboard") if user.role == "shop" else url_for("index")
            )

        flash("Invalid credentials")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ------------------ USER HOME ------------------
@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if current_user.role == "shop":
        return redirect(url_for("shop_dashboard"))

    if request.method == "POST":
        file = request.files.get("file")
        if not file or not allowed_file(file.filename):
            flash("Invalid image")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        prediction, confidence, treatment = predict_disease(path)

        record = History(
            user_id=current_user.id,
            filename=f"uploads/{filename}",
            prediction=prediction,
            confidence=confidence,
            fertilizer=", ".join(treatment["Fertilizer"]),
            pesticide=", ".join(treatment["Pesticide"]),
            organic=", ".join(treatment["Organic"])
        )
        db.session.add(record)
        db.session.commit()

        return render_template(
            "index.html",
            file=f"uploads/{filename}",
            prediction=prediction,
            confidence=confidence,
            fertilizer=treatment["Fertilizer"],
            pesticide=treatment["Pesticide"],
            organic=treatment["Organic"]
        )

    return render_template("index.html")

# ------------------ HISTORY ------------------
@app.route("/history")
@login_required
def history():
    if current_user.role != "user":
        return redirect(url_for("shop_dashboard"))

    records = History.query.filter_by(user_id=current_user.id).order_by(History.id.desc()).all()
    return render_template("history.html", records=records)


@app.route("/history/delete/<int:record_id>", methods=["POST"])
@login_required
def delete_record(record_id):
    record = History.query.get_or_404(record_id)
    if record.user_id != current_user.id:
        flash("Unauthorized")
        return redirect(url_for("history"))

    db.session.delete(record)
    db.session.commit()
    flash("Record deleted")
    return redirect(url_for("history"))

# ------------------ NEAREST SHOPS ------------------
@app.route("/shops")
@login_required
def shops():
    if current_user.role != "user":
        return redirect(url_for("shop_dashboard"))

    if not current_user.latitude or not current_user.longitude:
        flash("Location not available")
        return redirect(url_for("index"))

    nearby = []

    for shop in Shop.query.all():
        dist = haversine(
            current_user.latitude,
            current_user.longitude,
            shop.latitude,
            shop.longitude
        )

        if dist <= 15:
            shop.distance = round(dist, 2)
            nearby.append(shop)

    nearby.sort(key=lambda x: x.distance)

    return render_template("shops.html", shops=nearby)


# ------------------ USER VIEW SHOP PRODUCTS ------------------
@app.route("/shop/<int:shop_id>/products")
@login_required
def view_shop_products(shop_id):
    if current_user.role != "user":
        return redirect(url_for("index"))

    shop = Shop.query.get_or_404(shop_id)
    products = Product.query.filter_by(shop_id=shop.id).all()
    return render_template("shop_products_user.html", shop=shop, products=products)

# ------------------ SHOP DASHBOARD ------------------
@app.route("/shop/dashboard")
@login_required
def shop_dashboard():
    if current_user.role != "shop":
        return redirect(url_for("index"))

    shop = Shop.query.filter_by(owner_id=current_user.id).first()
    return render_template("shop_dashboard.html", shop=shop)

# ------------------ ADD SHOP ------------------
@app.route("/shop/add", methods=["GET", "POST"])
@login_required
def add_shop():
    if current_user.role != "shop":
        return redirect(url_for("index"))

    existing_shop = Shop.query.filter_by(owner_id=current_user.id).first()
    if existing_shop:
        flash("You already have a shop registered")
        return redirect(url_for("shop_dashboard"))

    if request.method == "POST":
        lat = request.form.get("latitude")
        lon = request.form.get("longitude")

        try:
            latitude = float(lat) if lat else None
            longitude = float(lon) if lon else None
        except ValueError:
            flash("Invalid location data")
            return redirect(url_for("add_shop"))  

        if latitude is None or longitude is None:
            flash("Location is required to add shop")
            return redirect(url_for("add_shop"))  

        shop = Shop(
            owner_id=current_user.id,
            shop_name=request.form["shop_name"],
            phone=request.form["phone"],
            address=request.form["address"],
            latitude=latitude,
            longitude=longitude
        )

        db.session.add(shop)
        db.session.commit()

        flash("Shop added successfully")
        return redirect(url_for("shop_dashboard"))

    return render_template("shop_add.html")


# ------------------ ADD PRODUCT ------------------
@app.route("/shop/products/add", methods=["GET", "POST"])
@login_required
def add_product():
    if current_user.role != "shop":
        return redirect(url_for("index"))

    shop = Shop.query.filter_by(owner_id=current_user.id).first()
    if not shop:
        flash("Add shop first")
        return redirect(url_for("shop_dashboard"))

    if request.method == "POST":
        product = Product(
            shop_id=shop.id,
            name=request.form["name"],
            category=request.form["category"],
            price=float(request.form["price"]),
            description=request.form["description"]
        )
        db.session.add(product)
        db.session.commit()
        flash("Product added")
        return redirect(url_for("shop_products"))

    return render_template("add_product.html")

# ------------------ SHOP VIEW PRODUCTS ------------------
@app.route("/shop/products")
@login_required
def shop_products():
    if current_user.role != "shop":
        return redirect(url_for("index"))

    shop = Shop.query.filter_by(owner_id=current_user.id).first()
    products = Product.query.filter_by(shop_id=shop.id).all()
    return render_template("shop_products.html", products=products)
# ================== CAMERA ==================
import base64, uuid
from io import BytesIO

@app.route("/camera_predict", methods=["POST"])
@login_required
def camera_predict():
    data = request.form.get("image")

    if not data:
        flash("Camera image not received")
        return redirect(url_for("index"))

    # Decode base64 image
    image_data = base64.b64decode(data.split(",")[1])
    filename = f"camera_{uuid.uuid4().hex}.jpg"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    with open(path, "wb") as f:
        f.write(image_data)

    prediction, confidence, treatment = predict_disease(path)

    # Save history
    record = History(
        user_id=current_user.id,
        filename=f"uploads/{filename}",
        prediction=prediction,
        confidence=confidence,
        fertilizer=", ".join(treatment["Fertilizer"]),
        pesticide=", ".join(treatment["Pesticide"]),
        organic=", ".join(treatment["Organic"])
    )
    db.session.add(record)
    db.session.commit()

    return render_template(
        "index.html",
        file=f"uploads/{filename}",
        prediction=prediction,
        confidence=confidence,
        fertilizer=treatment["Fertilizer"],
        pesticide=treatment["Pesticide"],
        organic=treatment["Organic"]
    )

# ------------------ RUN ------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)



