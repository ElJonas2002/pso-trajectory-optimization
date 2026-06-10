# 🦾 Energy-Efficient Trajectory Optimization with PSO

> Energy-aware motion strategy for a **2-DOF robotic manipulator**, using Particle Swarm Optimization (PSO) and cubic-spline trajectory planning to minimize energy consumption.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![MATLAB](https://img.shields.io/badge/MATLAB-0076A8?logo=mathworks&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)

## ✨ Highlights

| Metric | Result |
|---|---|
| Energy reduction | **26.7%** vs. a trapezoidal velocity-profile trajectory |
| Manipulator | **2 degrees of freedom** |
| Smoothness criterion | **Peak joint torque** |
| Planning method | Cubic spline interpolation + PSO |

## 🧠 How It Works

1. **Modeling** — The manipulator's dynamics are modeled to compute joint torques and energy consumption from the dynamic equations.
2. **Trajectory planning** — Cubic spline interpolation generates smooth joint trajectories, with peak joint torque used as the smoothness metric.
3. **Optimization** — Particle Swarm Optimization searches the trajectory parameter space to minimize total energy consumption.
4. **Validation** — Dynamics and trajectories are simulated in Python and MATLAB, with visualizations and performance analysis comparing the optimized path against a trapezoidal baseline.

## 📄 Publication

Presented at Trajectories 2026 Conference — *publication pending*.
<!-- Add a link to the paper / proceedings / preprint here once available. -->

## 🛠️ Tech Stack

`Python` · `MATLAB` · `NumPy` · `Particle Swarm Optimization` · `Cubic Splines`

## 🚀 Getting Started

```bash
git clone https://github.com/[YOUR_USERNAME]/pso-trajectory-optimization.git
cd pso-trajectory-optimization

pip install -r requirements.txt
python main.py        # runs the optimization and generates plots
```

## 📁 Repository Structure

```
pso-trajectory-optimization/
├── src/                 # Dynamics model, PSO, spline planner
├── results/             # Plots, energy-comparison figures
├── paper/               # Conference paper / figures
├── requirements.txt
└── README.md
```

## 👤 Author

**Jonathan Piña** — Robotics & AI Engineer
[LinkedIn]([YOUR_LINKEDIN]) · jonas.orlaineta02@gmail.com
