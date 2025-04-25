import streamlit as st
import os

# Title
st.title("Vector Compressed Sensing")

# Dropdown for algorithm
with st.sidebar:
    algo = st.selectbox(
        "Choose Algorithm:",
        options=["cvx", "softsense_jit", "steinsense"],
        format_func=lambda x: {
            "cvx": "Convex Optimization",
            "softsense_jit": "SoftSense",
            "steinsense": "SteinSense"
        }[x]
    )
    
    # Other selections
    N = st.selectbox("Choose number of vectors:", [100, 200, 300, 400, 500, 800, 1000, 1600, 2000, 5000])
    B = st.selectbox("Choose dimension of vectors:", [1, 3, 4, 5, 10, 20, 50])
    
    nonzero_dist = st.selectbox(
        "Choose distribution of nonzeros:",
        options=["standard normal", "binary_centered_p_0.5", "binary_random_sign", "gaussian_absolute",
                 "poisson_2", "poisson_hetero",
                 "mix_match", "binary_0.5", "exp_5"],
        format_func=lambda x: {
            "standard normal": "standard normal",
            "binary_centered_p_0.5": "scaled rademacher: ±0.5 w.p. 0.5",
            "binary_random_sign": "zero inflated binary: 0 w.p. 0.5, ±1 w.p. 0.25",
            "gaussian_absolute": "absolute standard normal",
            "poisson_2": "poisson(2)",
            "poisson_hetero": "poisson(j) in position j",
            "mix_match": "mash of different distributions",
            "binary_0.5": "bernoulli(0.5)",
            "exp_5": "exponential(scale = 5)"
        }[x]
    )

# Determine distribution tag
if nonzero_dist == "standard normal":
    dist_tag = ""
else:
    dist_tag = f"_{nonzero_dist.lower()}"

# Build file path
file_path = f"www/plots/paper_{algo}_N_{N}_B_{B}{dist_tag}.png"

# st.write(f"Trying to load: `{file_path}`")

if os.path.exists(file_path):
    st.image(file_path, use_container_width=True)
else:
    st.error(f"File does not exist: `{file_path}`")
