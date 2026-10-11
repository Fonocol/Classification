import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import pandas as pd

st.set_page_config(page_title="Classification MLP vs CNN", layout="centered")

st.title("Classification d'images : MLP vs CNN")

CIFAR10_CLASSES = [
    'Avion', 'Voiture', 'Oiseau', 'Chat', 'Cerf',
    'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion'
]

# Chargement des modèles
@st.cache_resource
def load_models():
    files = {
        "MLP MNIST": "mlp_mnist.keras",
        "CNN MNIST": "cnn_mnist.keras",
        "MLP CIFAR": "mlp_cifar.keras",
        "CNN CIFAR": "cnn_cifar.keras"
    }
    loaded = {}
    for name, path in files.items():
        try:
            loaded[name] = tf.keras.models.load_model(path)
        except Exception:
            loaded[name] = None
    return loaded

models = load_models()

# Résumé technique des modèles disponibles
st.subheader("Modèles chargés")
summary_rows = []
for name, m in models.items():
    if m is not None:
        summary_rows.append({
            "Modèle": name,
            "Paramètres": f"{m.count_params():,}",
            "Couches": len(m.layers),
            "Statut": "Prêt"
        })
    else:
        summary_rows.append({
            "Modèle": name,
            "Paramètres": "-",
            "Couches": "-",
            "Statut": "Fichier manquant"
        })
st.dataframe(pd.DataFrame(summary_rows).set_index("Modèle"), use_container_width=True)

st.write("---")

tab1, tab2 = st.tabs(["MNIST (Chiffres)", "CIFAR-10 (Objets)"])

# ----------------- SECTION MNIST -----------------
with tab1:
    st.subheader("Test sur MNIST")
    st.write("Dessinez un chiffre à tester :")

    from streamlit_drawable_canvas import st_canvas
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        return_image_data=True,
        key="canvas_mnist"
    )

    if st.button("Évaluer les modèles MNIST", key="btn_mnist"):
        if canvas_result is not None and canvas_result.image_data is not None:
            raw_data = canvas_result.image_data.astype("uint8")
            
            if np.max(raw_data[:, :, :3]) == 0:
                st.warning("La zone de dessin est vide.")
            else:
                img_pil = Image.fromarray(raw_data).convert("L")
                img_28 = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
                img_arr = np.array(img_28).astype("float32") / 255.0

                # Affichage de l'image d'entrée traitée
                st.write("**Image d'entrée reçue par les modèles (28×28 pixels) :**")
                st.image(img_28, width=140)

                col_mlp, col_cnn = st.columns(2)

                # MLP MNIST
                with col_mlp:
                    st.markdown("**MLP MNIST**")
                    if models["MLP MNIST"]:
                        inp = img_arr.reshape(1, 784)
                        t0 = time.time()
                        preds = models["MLP MNIST"].predict(inp, verbose=0)[0]
                        dt = (time.time() - t0) * 1000

                        st.write(f"- Prédiction : **{np.argmax(preds)}**")
                        st.write(f"- Confiance : **{np.max(preds)*100:.1f}%**")
                        st.write(f"- Latence : {dt:.1f} ms")
                        st.bar_chart(pd.DataFrame({"Proba": preds}, index=[str(i) for i in range(10)]))
                    else:
                        st.error("mlp_mnist.keras introuvable.")

                # CNN MNIST
                with col_cnn:
                    st.markdown("**CNN MNIST (LeNet-5)**")
                    if models["CNN MNIST"]:
                        inp = img_arr.reshape(1, 28, 28, 1)
                        t0 = time.time()
                        preds = models["CNN MNIST"].predict(inp, verbose=0)[0]
                        dt = (time.time() - t0) * 1000

                        st.write(f"- Prédiction : **{np.argmax(preds)}**")
                        st.write(f"- Confiance : **{np.max(preds)*100:.1f}%**")
                        st.write(f"- Latence : {dt:.1f} ms")
                        st.bar_chart(pd.DataFrame({"Proba": preds}, index=[str(i) for i in range(10)]))
                    else:
                        st.error("cnn_mnist.keras introuvable.")

# ----------------- SECTION CIFAR-10 -----------------
with tab2:
    st.subheader("Test sur CIFAR-10")
    uploaded_file = st.file_uploader("Image à analyser (PNG / JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_32 = img.resize((32, 32), Image.Resampling.BILINEAR)
        img_arr_cifar = np.array(img_32).astype("float32") / 255.0

        # Affichage côte à côte : image d'origine et image d'entrée 32x32
        st.write("**Images en entrée :**")
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            st.image(img, caption="Image originale téléversée", width=200)
        with col_in2:
            st.image(img_32, caption="Image redimensionnée (32×32) injectée dans les modèles", width=140)

        if st.button("Évaluer les modèles CIFAR", key="btn_cifar"):
            col_mlp_c, col_cnn_c = st.columns(2)

            # MLP CIFAR
            with col_mlp_c:
                st.markdown("**MLP CIFAR-10**")
                if models["MLP CIFAR"]:
                    inp = img_arr_cifar.reshape(1, 3072)
                    t0 = time.time()
                    preds = models["MLP CIFAR"].predict(inp, verbose=0)[0]
                    dt = (time.time() - t0) * 1000
                    idx = int(np.argmax(preds))

                    st.write(f"- Classe : **{CIFAR10_CLASSES[idx]}**")
                    st.write(f"- Confiance : **{preds[idx]*100:.1f}%**")
                    st.write(f"- Latence : {dt:.1f} ms")
                    st.bar_chart(pd.DataFrame({"Proba": preds}, index=CIFAR10_CLASSES))
                else:
                    st.error("mlp_cifar.keras introuvable.")

            # CNN CIFAR
            with col_cnn_c:
                st.markdown("**CNN CIFAR-10**")
                if models["CNN CIFAR"]:
                    inp = img_arr_cifar.reshape(1, 32, 32, 3)
                    t0 = time.time()
                    preds = models["CNN CIFAR"].predict(inp, verbose=0)[0]
                    dt = (time.time() - t0) * 1000
                    idx = int(np.argmax(preds))

                    st.write(f"- Classe : **{CIFAR10_CLASSES[idx]}**")
                    st.write(f"- Confiance : **{preds[idx]*100:.1f}%**")
                    st.write(f"- Latence : {dt:.1f} ms")
                    st.bar_chart(pd.DataFrame({"Proba": preds}, index=CIFAR10_CLASSES))
                else:
                    st.error("cnn_cifar.keras introuvable.")