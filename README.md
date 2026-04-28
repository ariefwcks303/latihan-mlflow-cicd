# 🏦 Credit Scoring Automation: Advanced MLOps Pipeline with MLflow & GitHub Actions

<p align="center">
  <img src="https://img.shields.io/github/workflow/status/ariefwcks303/latihan-mlflow-cicd/CI?label=Pipeline%# 🏦 Credit Scoring Automation: Advanced MLOps Pipeline with MLflow & GitHub Actions

<p align="center">
  <img src="https://img.shields.io/github/workflow/status/ariefwcks303/latihan-mlflow-cicd/CI?label=Pipeline%20Status&style=for-the-badge" alt="Pipeline Status">
  <img src="https://img.shields.io/badge/MLOps-MLflow-blue?style=for-the-badge" alt="MLOps MLflow">
  <img src="https://img.shields.io/badge/Automation-GitHub%20Actions-black?style=for-the-badge" alt="GitHub Actions">
  <img src="https://img.shields.io/badge/Storage-Git%20LFS-red?style=for-the-badge" alt="Git LFS">
</p>

## 📌 Project Overview

Project ini bukan sekadar melatih model prediksi. Ini adalah implementasi **Full-Stack MLOps Lifecycle** yang berfokus pada **Reproducibility**, **Automation**, dan **Observability**.

Tujuan utamanya adalah mengotomatisasi siklus pelatihan model klasifikasi untuk **Credit Scoring**, memastikan bahwa setiap perubahan kode atau parameter terlacak, dan model yang dihasilkan siap untuk dipublikasikan (_deploy-ready_).

---

## 🏗️ The MLOps Pipeline (Visual Workflow)

Di bawah ini adalah alur kerja otomatisasi yang berjalan setiap kali terjadi perubahan pada kode (git push):

```mermaid
graph TD
    A[Developer] -->|Push Code| B(GitHub Repository)
    B -->|Trigger Workflow| C{GitHub Actions Runner}
    subgraph "CI/CD Pipeline"
    C -->|1. Setup| D[Install Python & dependencies]
    D -->|2. Exec| E[Run MLflow Project (modelling.py)]
    subgraph "MLflow Execution"
    E -->|a. Train| F[Random Forest Classifier]
    E -->|b. Log| G[SQLite DB (Parameters, Metrics)]
    E -->|c. Save| H[Artifacts (Model .pkl)]
    end
    H -->|I/O| I[Git LFS Storage]
    end
    I -->|Sync| J(Update Repo)
    J -->|Results| B
    B -->|Monitor| K[Developer (mlflow ui)]
```
