import streamlit as st
from pmu_placer import UniversalPMUPlacer
import matplotlib.pyplot as plt

st.title("IEEE Bus PMU Placement Solver")

st.write("Select an IEEE bus system to solve for optimal PMU placement.")

ieee_bus = st.selectbox("Select IEEE Bus System", options=[14, 30, 57, 118, 300])

if st.button("Run Solver"):
    with st.spinner("Solving..."):
        solver = UniversalPMUPlacer(ieee_bus)
        solution = solver.solve()
        st.success(f"Optimal PMUs: {len(solution)} at buses {sorted(solution)}")

        # Visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        fig = solver.visualize(solution, fig=fig, ax=ax)
        st.pyplot(fig)
