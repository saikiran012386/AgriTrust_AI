"""
AgriTrust AI – Authentication Module
======================================
Lightweight session-state–based auth for the Streamlit demo.

Production Note:
  In a real deployment this would integrate with:
  • OAuth 2.0 / OpenID Connect (e.g., Keycloak, Auth0, Azure AD B2C)
  • JWT token validation on every API call
  • RBAC policies stored in the database with audit logging
  • MFA enforcement for Admin role
  The hardcoded credentials here exist solely for hackathon demo purposes.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Demo credential store
# In production: replace with a hashed-password lookup against a secure DB.
# ---------------------------------------------------------------------------
USERS: dict[str, dict] = {
    "admin": {
        "password": "admin123",
        "role":     "Admin",
        "display":  "Dr. Rajan Mehta",
        "branch":   "Head Office – New Delhi",
    },
    "officer": {
        "password": "officer123",
        "role":     "Loan Officer",
        "display":  "Ms. Priya Sharma",
        "branch":   "Regional Branch – Pune",
    },
    "demo": {
        "password": "demo",
        "role":     "Loan Officer",
        "display":  "Demo User",
        "branch":   "Demo Branch",
    },
    "rahul": {
        "password": "93477",
        "role":     "farmerr",
        "display":  "rahul",
        "branch":   "siddipet",
    },
     "vignesh": {
        "password": "93477",
        "role":     "devlop+er",
        "display":  "vignesh",
        "branch":   "hyd",
    },
}


def check_credentials(username: str, password: str) -> dict | None:
    """Return user info dict on success, None on failure."""
    user = USERS.get(username.lower().strip())
    if user and user["password"] == password:
        return user
    return None


def login_ui() -> None:
    """Render the login screen.  Sets st.session_state on success."""
    # ── Page chrome ──────────────────────────────────────────────────────────
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

      .login-shell {
        max-width: 420px;
        margin: 60px auto 0;
        padding: 48px 40px 40px;
        background: linear-gradient(155deg, #0d2137 0%, #0a3d2e 100%);
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 32px 80px rgba(0,0,0,0.45);
      }
      .login-logo {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        color: #4ade80;
        text-align: center;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
      }
      .login-sub {
        text-align: center;
        color: rgba(255,255,255,0.45);
        font-size: 0.78rem;
        margin-bottom: 36px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
      }
      .demo-hint {
        background: rgba(74,222,128,0.08);
        border: 1px solid rgba(74,222,128,0.2);
        border-radius: 10px;
        padding: 14px 16px;
        margin-top: 20px;
        font-size: 0.8rem;
        color: rgba(255,255,255,0.6);
        line-height: 1.7;
      }
      .demo-hint strong { color: #4ade80; }
    </style>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 2.2, 1])
    with col_m:
        st.markdown('<div class="login-logo">🌾 AgriTrust AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">Rural Credit Intelligence Platform</div>', unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        if st.button("Sign In →", use_container_width=True, type="primary"):
            user = check_credentials(username, password)
            if user:
                st.session_state["authenticated"] = True
                st.session_state["user"]           = user
                st.session_state["username"]       = username
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")

        st.markdown("""
        <div class="demo-hint">
          <strong>Demo Credentials</strong><br>
          Admin &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ admin / admin123<br>
          Officer → officer / officer123
        </div>
        """, unsafe_allow_html=True)


def require_auth() -> dict:
    """
    Call at the top of app.py.
    Returns user dict if authenticated, else renders login and stops.
    """
    if not st.session_state.get("authenticated"):
        login_ui()
        st.stop()
    return st.session_state["user"]


def logout() -> None:
    """Clear session state and trigger rerun."""
    for key in ["authenticated", "user", "username"]:
        st.session_state.pop(key, None)
    st.rerun()
