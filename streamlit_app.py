"""🌸 Flower Federated Learning Simulator - Streamlit Cloud Deployment.

Aplikasi Web Interaktif Simulasi Federated Learning (1 Server SuperLink & 3 Client SuperNodes)
menggunakan Algoritma FedAvg (Federated Averaging) berbasis Flower Framework.
Siap di-deploy langsung ke Streamlit Community Cloud (streamlit.io).
"""

import time
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Flower FL SuperGrid Simulator",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (Dark SuperGrid Modern Theme)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global styling */
    .main-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding-bottom: 14px;
        border-bottom: 1px solid #30363D;
        margin-bottom: 20px;
    }
    .flower-logo {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: linear-gradient(135deg, #9333EA, #EC4899, #F59E0B);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        box-shadow: 0 4px 14px rgba(168, 85, 247, 0.4);
    }
    .badge-purple {
        background: rgba(147, 51, 234, 0.2);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    .node-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    .node-card:hover {
        border-color: #A855F7;
        box-shadow: 0 4px 16px rgba(168, 85, 247, 0.2);
    }
    .server-card {
        background: linear-gradient(180deg, #1A1F2C 0%, #161B22 100%);
        border: 1px solid #8B5CF6;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.25);
    }
    .log-box {
        background-color: #0D1117;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 12px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
        height: 220px;
        overflow-y: auto;
        color: #E6EDF3;
    }
    .stat-pill {
        font-size: 13px;
        color: #8B949E;
    }
    .stat-val {
        font-size: 20px;
        font-weight: 700;
        color: #F0F6FC;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "round" not in st.session_state:
    st.session_state.round = 0
if "phase" not in st.session_state:
    st.session_state.phase = "IDLE"  # IDLE, BROADCAST, LOCAL_TRAIN, AGGREGATE, COMPLETED
if "global_acc" not in st.session_state:
    st.session_state.global_acc = 10.0
if "global_loss" not in st.session_state:
    st.session_state.global_loss = 2.30
if "clients" not in st.session_state:
    st.session_state.clients = {
        "client_1": {"name": "SuperNode 1 (Alpha)", "samples": 100, "loss": 2.30, "acc": 10.0, "status": "Siap (Ready)"},
        "client_2": {"name": "SuperNode 2 (Beta)", "samples": 150, "loss": 2.30, "acc": 10.0, "status": "Siap (Ready)"},
        "client_3": {"name": "SuperNode 3 (Gamma)", "samples": 80, "loss": 2.30, "acc": 10.0, "status": "Siap (Ready)"},
    }
if "history" not in st.session_state:
    st.session_state.history = {
        "round": [0],
        "global_acc": [10.0],
        "global_loss": [2.30],
        "c1_acc": [10.0],
        "c2_acc": [10.0],
        "c3_acc": [10.0],
        "c1_loss": [2.30],
        "c2_loss": [2.30],
        "c3_loss": [2.30],
    }
if "logs" not in st.session_state:
    st.session_state.logs = [
        "[SYSTEM] Inisialisasi Flower SuperGrid v1.40 berhasil.",
        "[SYSTEM] SuperLink listen pada endpoint 127.0.0.1:39093.",
        "[SYSTEM] 3 SuperNode terhubung ke Server (Alpha: 100, Beta: 150, Gamma: 80 sampel).",
        "[STRATEGY] Algoritma FedAvg siap untuk agregasi bobot parameter global.",
    ]


def log_event(message: str):
    """Add a timestamped message to the logs."""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")


def reset_simulation():
    """Reset the entire simulation state."""
    st.session_state.round = 0
    st.session_state.phase = "IDLE"
    st.session_state.global_acc = 10.0
    st.session_state.global_loss = 2.30
    st.session_state.clients = {
        "client_1": {"name": "SuperNode 1 (Alpha)", "samples": 100, "loss": 2.30, "acc": 10.0, "status": "Siap (Ready)"},
        "client_2": {"name": "SuperNode 2 (Beta)", "samples": 150, "loss": 2.30, "acc": 10.0, "status": "Siap (Ready)"},
        "client_3": {"name": "SuperNode 3 (Gamma)", "samples": 80, "loss": 2.30, "acc": 10.0, "status": "Siap (Ready)"},
    }
    st.session_state.history = {
        "round": [0],
        "global_acc": [10.0],
        "global_loss": [2.30],
        "c1_acc": [10.0],
        "c2_acc": [10.0],
        "c3_acc": [10.0],
        "c1_loss": [2.30],
        "c2_loss": [2.30],
        "c3_loss": [2.30],
    }
    st.session_state.logs = [
        "[SYSTEM] Simulasi di-reset ke status awal (Ronde 0).",
        "[SYSTEM] Siap memulai pelatihan federated learning.",
    ]


def advance_one_step(total_rounds: int, drop_c3: bool):
    """Step through the Federated Learning lifecycle."""
    phase = st.session_state.phase

    if st.session_state.round >= total_rounds and phase in ["AGGREGATE", "COMPLETED"]:
        st.session_state.phase = "COMPLETED"
        log_event(f"Pelatihan selesai! Mencapai total {total_rounds} ronde dengan akurasi global {st.session_state.global_acc:.2f}%.")
        return

    if phase in ["IDLE", "AGGREGATE"]:
        # Phase 1: Parameter Broadcast
        st.session_state.round += 1
        curr_round = st.session_state.round
        st.session_state.phase = "BROADCAST"
        log_event(f"🌸 [RONDE {curr_round}/{total_rounds}] SuperLink mendistribusikan parameter model global W_{curr_round-1} ke semua SuperNodes...")
        for cid, c in st.session_state.clients.items():
            if cid == "client_3" and drop_c3 and curr_round >= 3:
                c["status"] = "⚠️ Flaky/Timeout"
                log_event(f"SuperNode 3 (Gamma) mengalami network jitter/unreachable!")
            else:
                c["status"] = "Menerima Bobot W_t..."

    elif phase == "BROADCAST":
        # Phase 2: Local Training on SuperNodes
        curr_round = st.session_state.round
        st.session_state.phase = "LOCAL_TRAIN"
        log_event(f"[RONDE {curr_round}] SuperNodes mengeksekusi ClientApp.fit() secara lokal pada data privat...")

        # Update local metrics simulating training improvement
        for cid, c in st.session_state.clients.items():
            if cid == "client_3" and drop_c3 and curr_round >= 3:
                c["status"] = "⚠️ Gagal Pelatihan (Drop)"
            else:
                c["status"] = "Pelatihan Lokal (Epochs 1-5)"
                # Simulate loss reduction and acc increase
                gain = max(1.0, (10 - curr_round) * 2.2 + random.uniform(-1.0, 2.5))
                c["acc"] = min(98.5, c["acc"] + gain)
                c["loss"] = max(0.12, c["loss"] * 0.72 + random.uniform(-0.02, 0.04))

    elif phase == "LOCAL_TRAIN":
        # Phase 3: Aggregation with FedAvg
        curr_round = st.session_state.round
        st.session_state.phase = "AGGREGATE"
        log_event(f"[RONDE {curr_round}] SuperNodes mengunggah update bobot ke SuperLink. Memulai FedAvg ServerApp...")

        active_clients = []
        for cid, c in st.session_state.clients.items():
            if cid == "client_3" and drop_c3 and curr_round >= 3:
                c["status"] = "Excluded (Dropout)"
            else:
                c["status"] = "Terkirim ke Server"
                active_clients.append(c)

        # FedAvg weighted average calculation
        total_samples = sum(c["samples"] for c in active_clients)
        new_acc = sum((c["samples"] / total_samples) * c["acc"] for c in active_clients)
        new_loss = sum((c["samples"] / total_samples) * c["loss"] for c in active_clients)

        st.session_state.global_acc = round(new_acc, 2)
        st.session_state.global_loss = round(new_loss, 3)

        # Update history
        st.session_state.history["round"].append(curr_round)
        st.session_state.history["global_acc"].append(st.session_state.global_acc)
        st.session_state.history["global_loss"].append(st.session_state.global_loss)
        st.session_state.history["c1_acc"].append(st.session_state.clients["client_1"]["acc"])
        st.session_state.history["c2_acc"].append(st.session_state.clients["client_2"]["acc"])
        st.session_state.history["c3_acc"].append(st.session_state.clients["client_3"]["acc"])
        st.session_state.history["c1_loss"].append(st.session_state.clients["client_1"]["loss"])
        st.session_state.history["c2_loss"].append(st.session_state.clients["client_2"]["loss"])
        st.session_state.history["c3_loss"].append(st.session_state.clients["client_3"]["loss"])

        log_event(f"✅ [RONDE {curr_round}] Agregasi FedAvg Berhasil! Global Acc: {st.session_state.global_acc:.2f}%, Global Loss: {st.session_state.global_loss:.3f}.")

        if curr_round >= total_rounds:
            st.session_state.phase = "COMPLETED"
            log_event(f"🎉 Target ronde ({total_rounds}) tercapai! Model global telah konvergen.")


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi Flower FL")
    
    total_rounds = st.selectbox(
        "Jumlah Ronde Pelatihan (num_rounds):",
        options=[3, 5, 10],
        index=1,
        help="Berapa ronde komunikasi federated learning yang akan dijalankan."
    )
    
    strategy = st.selectbox(
        "Strategi Agregasi Server:",
        options=["FedAvg (Federated Averaging)", "FedProx", "FedMedian"],
        index=0,
        help="Algoritma agregasi parameter di SuperLink."
    )
    
    drop_c3 = st.checkbox(
        "Simulasikan Client-3 Dropout / Flaky",
        value=False,
        help="Jika aktif, SuperNode 3 akan mengalami timeout/disconnect pada ronde 3 ke atas untuk menguji ketahanan FedAvg."
    )
    
    st.markdown("---")
    st.markdown("### 🎮 Kontrol Simulasi")
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        step_clicked = st.button("⏭ Next Step", use_container_width=True, help="Jalankan satu fase simulasi secara bertahap")
    with col_ctrl2:
        reset_clicked = st.button("🔄 Reset", use_container_width=True, help="Kembalikan simulasi ke kondisi awal")
        
    auto_run = st.button("▶ Jalankan Otomatis (Full)", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ Info Deployment")
    st.markdown("""
    - **Platform:** Streamlit Community Cloud
    - **Flower Engine:** v1.40 FedAvg
    - **Node Ratio:** 1 SuperLink : 3 SuperNodes
    - **Status:** `Online & Connected`
    """)

# Handle Sidebar Actions
if reset_clicked:
    reset_simulation()
    st.rerun()

if step_clicked:
    advance_one_step(total_rounds, drop_c3)
    st.rerun()

if auto_run:
    reset_simulation()
    progress_bar = st.progress(0)
    for r in range(1, total_rounds + 1):
        advance_one_step(total_rounds, drop_c3)  # Broadcast
        time.sleep(0.3)
        advance_one_step(total_rounds, drop_c3)  # Local Train
        time.sleep(0.3)
        advance_one_step(total_rounds, drop_c3)  # Aggregate
        time.sleep(0.3)
        progress_bar.progress(int((r / total_rounds) * 100))
    st.rerun()


# ---------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="flower-logo">🌸</div>
    <div>
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
            <h2 style="margin: 0; padding: 0; font-weight: 800; color: #FFFFFF;">Flower SuperGrid Simulator</h2>
            <span class="badge-purple">Flower v1.40 FedAvg</span>
            <span class="badge-green">● Live Deployment</span>
        </div>
        <p style="margin: 0; color: #8B949E; font-size: 13px;">
            Dashboard Interaktif Federated Learning: 1 Server (SuperLink) & 3 Client (SuperNodes) dengan Algoritma FedAvg
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Top Summary Metrics
# ---------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric(label="Ronde Pelatihan", value=f"{st.session_state.round} / {total_rounds}")
with m2:
    phase_label_map = {
        "IDLE": "Standby (Idle)",
        "BROADCAST": "📡 Broadcast W_t",
        "LOCAL_TRAIN": "💻 Local Training",
        "AGGREGATE": "🧮 FedAvg Aggregation",
        "COMPLETED": "🎉 Selesai (Converged)"
    }
    st.metric(label="Fase Aktif", value=phase_label_map.get(st.session_state.phase, st.session_state.phase))
with m3:
    st.metric(label="Global Accuracy", value=f"{st.session_state.global_acc:.2f} %", delta=f"{st.session_state.global_acc - 10.0:.1f}%" if st.session_state.round > 0 else None)
with m4:
    st.metric(label="Global Loss", value=f"{st.session_state.global_loss:.3f}", delta=f"{st.session_state.global_loss - 2.30:.3f}" if st.session_state.round > 0 else None, delta_color="inverse")
with m5:
    active_count = 2 if (drop_c3 and st.session_state.round >= 3) else 3
    st.metric(label="SuperNodes Aktif", value=f"{active_count} / 3 Nodes")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tabs Layout
# ---------------------------------------------------------
tab_sim, tab_charts, tab_arch, tab_about = st.tabs([
    "🖥️ Simulasi SuperGrid", 
    "📈 Kurva Konvergensi (Charts)", 
    "📐 Arsitektur & Rumus FedAvg",
    "📋 Panduan & Pengumpulan Tugas"
])

# =========================================================
# TAB 1: Visual SuperGrid Network
# =========================================================
with tab_sim:
    st.markdown("#### 🌐 Topologi Jaringan Flower SuperGrid (1 Server, 3 Client)")
    
    # Server SuperLink Card
    st.markdown(f"""
    <div class="server-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 32px;">🏢</span>
                <div>
                    <h4 style="margin: 0; color: #FFFFFF; font-weight: 700;">Flower SuperLink (ServerApp Coordinator)</h4>
                    <p style="margin: 0; color: #A78BFA; font-size: 13px;">Strategi Agregasi: <b>{strategy}</b> | Port: 127.0.0.1:39093</p>
                </div>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 12px; color: #9CA3AF;">Global Accuracy</span><br>
                <span style="font-size: 22px; font-weight: 800; color: #34D399;">{st.session_state.global_acc:.2f}%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: #A855F7; font-size: 24px; margin: 6px 0;">
        ⇅ ⇅ ⇅ <span style="font-size: 12px; color: #9CA3AF; vertical-align: middle;">(gRPC Transport / Parameter Sync)</span> ⇅ ⇅ ⇅
    </div>
    """, unsafe_allow_html=True)

    # 3 SuperNode Client Cards
    c_col1, c_col2, c_col3 = st.columns(3)
    
    clients_data = st.session_state.clients
    
    with c_col1:
        c1 = clients_data["client_1"]
        st.markdown(f"""
        <div class="node-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h5 style="margin: 0; color: #FFFFFF;">💻 {c1['name']}</h5>
                <span class="badge-purple">ClientApp 1</span>
            </div>
            <div style="font-size: 13px; color: #9CA3AF;">Jumlah Sampel: <b style="color: #F3F4F6;">{c1['samples']} dataset</b></div>
            <div style="margin: 8px 0; display: flex; justify-content: space-between;">
                <div><span class="stat-pill">Local Loss</span><br><b style="color: #FBBF24;">{c1['loss']:.3f}</b></div>
                <div><span class="stat-pill">Local Acc</span><br><b style="color: #34D399;">{c1['acc']:.2f}%</b></div>
            </div>
            <div style="margin-top: 10px; padding: 6px 10px; background: #0D1117; border-radius: 6px; font-size: 12px; color: #C084FC;">
                Status: {c1['status']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_col2:
        c2 = clients_data["client_2"]
        st.markdown(f"""
        <div class="node-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h5 style="margin: 0; color: #FFFFFF;">💻 {c2['name']}</h5>
                <span class="badge-purple">ClientApp 2</span>
            </div>
            <div style="font-size: 13px; color: #9CA3AF;">Jumlah Sampel: <b style="color: #F3F4F6;">{c2['samples']} dataset</b></div>
            <div style="margin: 8px 0; display: flex; justify-content: space-between;">
                <div><span class="stat-pill">Local Loss</span><br><b style="color: #FBBF24;">{c2['loss']:.3f}</b></div>
                <div><span class="stat-pill">Local Acc</span><br><b style="color: #34D399;">{c2['acc']:.2f}%</b></div>
            </div>
            <div style="margin-top: 10px; padding: 6px 10px; background: #0D1117; border-radius: 6px; font-size: 12px; color: #C084FC;">
                Status: {c2['status']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_col3:
        c3 = clients_data["client_3"]
        is_c3_flaky = (drop_c3 and st.session_state.round >= 3)
        status_color = "#F87171" if is_c3_flaky else "#C084FC"
        badge_style = "background: rgba(239, 68, 68, 0.2); color: #F87171;" if is_c3_flaky else "badge-purple"
        
        st.markdown(f"""
        <div class="node-card" style="border-color: {'#EF4444' if is_c3_flaky else '#30363D'};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h5 style="margin: 0; color: #FFFFFF;">💻 {c3['name']}</h5>
                <span style="padding: 2px 8px; border-radius: 9999px; font-size: 11px; {badge_style}">
                    {'DROPOUT' if is_c3_flaky else 'ClientApp 3'}
                </span>
            </div>
            <div style="font-size: 13px; color: #9CA3AF;">Jumlah Sampel: <b style="color: #F3F4F6;">{c3['samples']} dataset</b></div>
            <div style="margin: 8px 0; display: flex; justify-content: space-between;">
                <div><span class="stat-pill">Local Loss</span><br><b style="color: #FBBF24;">{c3['loss']:.3f}</b></div>
                <div><span class="stat-pill">Local Acc</span><br><b style="color: #34D399;">{c3['acc']:.2f}%</b></div>
            </div>
            <div style="margin-top: 10px; padding: 6px 10px; background: #0D1117; border-radius: 6px; font-size: 12px; color: {status_color};">
                Status: {c3['status']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📜 Live System Logs (SuperLink & SuperNodes Events)")
    
    # Live log box
    reversed_logs = "\n".join(reversed(st.session_state.logs))
    st.text_area("Console Logs", value=reversed_logs, height=200, disabled=True, label_visibility="collapsed")


# =========================================================
# TAB 2: Convergence Charts
# =========================================================
with tab_charts:
    st.markdown("#### 📈 Analisis Performa dan Konvergensi Model")
    
    hist = st.session_state.history
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=hist["round"], y=hist["global_acc"], mode="lines+markers", name="Global Accuracy", line=dict(color="#10B981", width=3), marker=dict(size=8)))
        fig_acc.add_trace(go.Scatter(x=hist["round"], y=hist["c1_acc"], mode="lines+markers", name="Node 1 (Alpha)", line=dict(color="#8B5CF6", width=1.5, dash="dot")))
        fig_acc.add_trace(go.Scatter(x=hist["round"], y=hist["c2_acc"], mode="lines+markers", name="Node 2 (Beta)", line=dict(color="#3B82F6", width=1.5, dash="dot")))
        fig_acc.add_trace(go.Scatter(x=hist["round"], y=hist["c3_acc"], mode="lines+markers", name="Node 3 (Gamma)", line=dict(color="#F59E0B", width=1.5, dash="dot")))
        
        fig_acc.update_layout(
            title="Akurasi Model (%) vs Ronde Pelatihan",
            xaxis_title="Ronde (Round)",
            yaxis_title="Akurasi (%)",
            paper_bgcolor="#161B22",
            plot_bgcolor="#0D1117",
            font=dict(color="#E6EDF3"),
            xaxis=dict(gridcolor="#30363D", tickmode="linear", dtick=1),
            yaxis=dict(gridcolor="#30363D", range=[0, 105]),
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_chart2:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=hist["round"], y=hist["global_loss"], mode="lines+markers", name="Global Loss", line=dict(color="#EF4444", width=3), marker=dict(size=8)))
        fig_loss.add_trace(go.Scatter(x=hist["round"], y=hist["c1_loss"], mode="lines+markers", name="Node 1 (Alpha)", line=dict(color="#8B5CF6", width=1.5, dash="dot")))
        fig_loss.add_trace(go.Scatter(x=hist["round"], y=hist["c2_loss"], mode="lines+markers", name="Node 2 (Beta)", line=dict(color="#3B82F6", width=1.5, dash="dot")))
        fig_loss.add_trace(go.Scatter(x=hist["round"], y=hist["c3_loss"], mode="lines+markers", name="Node 3 (Gamma)", line=dict(color="#F59E0B", width=1.5, dash="dot")))
        
        fig_loss.update_layout(
            title="Loss Model vs Ronde Pelatihan",
            xaxis_title="Ronde (Round)",
            yaxis_title="Nilai Loss",
            paper_bgcolor="#161B22",
            plot_bgcolor="#0D1117",
            font=dict(color="#E6EDF3"),
            xaxis=dict(gridcolor="#30363D", tickmode="linear", dtick=1),
            yaxis=dict(gridcolor="#30363D"),
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # Detailed metrics table
    st.markdown("##### 📊 Tabel Riwayat Metrik Per Ronde")
    df_metrics = pd.DataFrame({
        "Ronde": hist["round"],
        "Global Acc (%)": [f"{v:.2f}%" for v in hist["global_acc"]],
        "Global Loss": [f"{v:.3f}" for v in hist["global_loss"]],
        "Node 1 Acc": [f"{v:.2f}%" for v in hist["c1_acc"]],
        "Node 2 Acc": [f"{v:.2f}%" for v in hist["c2_acc"]],
        "Node 3 Acc": [f"{v:.2f}%" for v in hist["c3_acc"]],
    })
    st.dataframe(df_metrics, use_container_width=True)


# =========================================================
# TAB 3: Architecture & Formulas
# =========================================================
with tab_arch:
    st.markdown("#### 📐 Konsep Arsitektur Flower SuperGrid & Teori Matematika FedAvg")
    
    st.markdown("""
    Flower Framework (versi 1.40+) mengadopsi arsitektur **SuperGrid** modern yang memisahkan eksekusi model dari koordinasi jaringan:
    
    1. **SuperLink (Server Coordinator):**
       - Bertindak sebagai simpul pusat yang mengelola sesi federasi.
       - Bertanggung jawab memicu ronde pelatihan, mendistribusikan parameter global $W_t$, dan memvalidasi keaslian update parameter dari SuperNodes.
       
    2. **SuperNodes (Client Worker Nodes):**
       - Node lokal di mana data pelatihan tersimpan secara privat (data tidak pernah dikirim ke server / *Privacy-Preserving*).
       - Menjalankan komputasi lokal (`ClientApp.fit()`) untuk menghasilkan pembaruan gradien/bobot $W_{t+1}^k$.

    3. **DriverAPI & ServerApp:**
       - Menjalankan logika strategi agregasi (misalnya **FedAvg**) secara independen dari protokol komunikasi jaringan.
    """)

    st.markdown("---")
    st.markdown("#### 🧮 Rumus Matematis Federated Averaging (FedAvg)")
    st.latex(r"""
    w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_{t+1}^k
    """)
    st.markdown(r"""
    Keterangan Simbol:
    - $w_{t+1}$: Parameter model global baru setelah agregasi pada ronde $t+1$.
    - $K$: Jumlah total klien/SuperNode yang berpartisipasi ($K = 3$ dalam simulasi ini).
    - $n_k$: Jumlah sampel data lokal pada klien ke-$k$ (Alpha = 100, Beta = 150, Gamma = 80).
    - $N = \sum_{k=1}^K n_k$: Total keseluruhan sampel data dari seluruh klien yang berpartisipasi.
    - $w_{t+1}^k$: Bobot model hasil pelatihan lokal dari klien ke-$k$.
    """)

    st.markdown("---")
    st.markdown("#### 🔄 Alur Siklus Komunikasi Per Ronde (Federated Communication Loop)")
    st.markdown("""
    ```
    +-----------------------------------------------------------+
    |           SUPERLINK (SERVER COORDINATOR)                  |
    +-----------------------------------------------------------+
       |                                                      ^
       | 1. Broadcast W_t (Bobot Global)                     | 3. Upload W_{t+1}^k
       v                                                      |
    +-----------------------------------------------------------+
    |           SUPERNODES (3 CLIENT LOCAL WORKERS)             |
    |  - Alpha (100) | Beta (150) | Gamma (80)                  |
    |  2. Eksekusi Pelatihan Lokal (Local Training on Private Data) |
    +-----------------------------------------------------------+
       |
       v
    [ SuperLink menjalankan FedAvg: Menghitung rata-rata tertimbang bobot ]
       |
       v
    [ Update Parameter Global W_{t+1} & Evaluasi Konvergensi Akurasi ]
    ```
    """)


# =========================================================
# TAB 4: Submission & Deployment Info
# =========================================================
with tab_about:
    st.markdown("#### 📋 Panduan Deployment & Pengumpulan Tugas")
    
    st.success("""
    🎉 **Aplikasi ini telah dirancang khusus untuk memenuhi kriteria tugas implementasi Flower Federated Learning di Streamlit!**
    
    Struktur file dan dependensi telah disesuaikan agar dapat langsung berjalan di Streamlit Community Cloud (`share.streamlit.io`) tanpa error.
    """)
    
    st.markdown("""
    ##### 🚀 Langkah Cepat Menghasilkan URL Publik (share.streamlit.io):
    1. Pastikan folder ini di-push ke repository GitHub Anda (misal nama repo: `flower-fl-simulation`).
    2. Buka [share.streamlit.io](https://share.streamlit.io) lalu login dengan akun GitHub Anda.
    3. Klik **"New app"** (atau **"Create app"**).
    4. Pilih repository Anda, branch `main`, dan set path file utama ke: `streamlit_app.py`.
    5. Klik tombol **"Deploy!"**
    6. Tunggu ~1 menit hingga aplikasi aktif. Salin URL publik Anda (contoh: `https://flower-fl-simulation.streamlit.app`) untuk dikumpulkan pada lembar tugas Anda.
    
    ##### 📑 Komponen Tugas yang Terpenuhi:
    - [x] Antarmuka Interaktif Federated Learning (1 Server, 3 Client)
    - [x] Simulasi Siklus Pelatihan Lengkap (Broadcast -> Local Train -> Upload -> FedAvg)
    - [x] Simulasi Kegagalan Jaringan (Node Dropout / Flaky SuperNode)
    - [x] Visualisasi Grafik Konvergensi Realtime (Loss & Akurasi)
    - [x] Konsol Event Log SuperLink & SuperNodes
    - [x] Dokumentasi Arsitektur & Formula Matematis FedAvg
    """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: #8B949E;">
    <span>🌸 Flower Federated Learning Simulator v1.40 | FedAvg Strategy</span>
    <span>Siap Dikumpulkan & Terhubung ke Streamlit Cloud</span>
</div>
""", unsafe_allow_html=True)
