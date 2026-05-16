"""
Email Service — OTP Delivery via Resend (HTTP API)
=====================================================
Sends OTP verification emails using Resend's HTTP API.
This works on all hosting platforms (including Render free tier)
because it uses HTTPS instead of SMTP.

Setup:
    1. Sign up at https://resend.com (free, no credit card)
    2. Get your API key from the dashboard
    3. Set RESEND_API_KEY environment variable on Render
"""

import os
import logging

logger = logging.getLogger(__name__)

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "Quant Engine <onboarding@resend.dev>")


def _build_otp_html(otp: str, recipient_email: str) -> str:
    """Build a professional HTML email template for OTP delivery."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin:0; padding:0; background-color:#0d1117; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0d1117; padding: 40px 0;">
            <tr>
                <td align="center">
                    <table width="480" cellpadding="0" cellspacing="0" style="background-color:#161b22; border-radius:16px; border:1px solid #30363d; overflow:hidden;">
                        <!-- Header -->
                        <tr>
                            <td style="background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%); padding: 32px; text-align:center;">
                                <div style="font-size: 28px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">
                                    Quant Engine
                                </div>
                                <div style="font-size: 14px; color: rgba(255,255,255,0.85); margin-top: 6px;">
                                    Derivatives Pricing &amp; Risk Platform
                                </div>
                            </td>
                        </tr>
                        <!-- Body -->
                        <tr>
                            <td style="padding: 40px 32px;">
                                <h2 style="color:#c9d1d9; font-size:20px; margin:0 0 8px 0;">
                                    Verify your email
                                </h2>
                                <p style="color:#8b949e; font-size:14px; line-height:1.6; margin:0 0 28px 0;">
                                    Use the verification code below to complete your registration. This code expires in <strong style="color:#c9d1d9;">5 minutes</strong>.
                                </p>
                                <!-- OTP Box -->
                                <div style="background:#0d1117; border:1px solid #30363d; border-radius:12px; padding:24px; text-align:center; margin-bottom:28px;">
                                    <div style="font-size:36px; font-weight:700; letter-spacing:12px; color:#58a6ff; font-family: 'Courier New', monospace;">
                                        {otp}
                                    </div>
                                </div>
                                <p style="color:#8b949e; font-size:13px; line-height:1.5; margin:0;">
                                    If you didn't request this code, you can safely ignore this email.
                                </p>
                            </td>
                        </tr>
                        <!-- Footer -->
                        <tr>
                            <td style="padding: 20px 32px; border-top: 1px solid #30363d;">
                                <p style="color:#484f58; font-size:12px; margin:0; text-align:center;">
                                    Sent to {recipient_email} &mdash; Quant Engine Platform
                                </p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """


def send_otp_email(recipient: str, otp: str) -> bool:
    """
    Send the OTP verification email via Resend HTTP API.

    Returns True if sent successfully, False otherwise.
    Falls back to logging the OTP if Resend API key is not configured.
    """
    if not RESEND_API_KEY:
        logger.warning(
            f"RESEND_API_KEY not configured. OTP for {recipient}: {otp} "
            "(Set RESEND_API_KEY environment variable to enable email sending)"
        )
        return True  # Return True so the flow continues in dev

    import requests as req_lib

    try:
        resp = req_lib.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": "QuantEngine/3.0",
            },
            json={
                "from": RESEND_FROM_EMAIL,
                "to": [recipient],
                "subject": f"Your Quant Engine Verification Code: {otp}",
                "html": _build_otp_html(otp, recipient),
            },
            timeout=10,
        )

        if resp.status_code in (200, 201):
            result = resp.json()
            logger.info(f"OTP email sent to {recipient} via Resend (id={result.get('id', 'unknown')})")
            return True
        else:
            logger.error(f"Resend API error {resp.status_code} for {recipient}: {resp.text}")
            return False

    except Exception as e:
        logger.error(f"Failed to send OTP email to {recipient}: {e}")
        return False
