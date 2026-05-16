"""
Email Service — OTP Delivery via SMTP
=========================================
Sends a beautifully formatted OTP verification email using
standard Python smtplib. Falls back to logging the OTP if
SMTP is not configured.

Configure via environment variables:
    SMTP_SERVER   — e.g. smtp.gmail.com
    SMTP_PORT     — e.g. 587
    SMTP_USERNAME — e.g. epsilonenterprise7@gmail.com
    SMTP_PASSWORD — Gmail App Password (NOT your login password)
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

# SMTP configuration from environment
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Quant Engine Platform")


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
                                    Q Quant Engine
                                </div>
                                <div style="font-size: 14px; color: rgba(255,255,255,0.85); margin-top: 6px;">
                                    Derivatives Pricing & Risk Platform
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
                                    If you didn't request this code, you can safely ignore this email. Someone may have entered your email address by mistake.
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
    Send the OTP verification email to the recipient.
    
    Returns True if sent successfully, False otherwise.
    Falls back to logging the OTP if SMTP is not configured.
    """
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        logger.warning(
            f"SMTP not configured. OTP for {recipient}: {otp} "
            "(Set SMTP_USERNAME and SMTP_PASSWORD environment variables)"
        )
        return True  # Return True so the flow continues in dev

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Your Quant Engine Verification Code: {otp}"
        msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_USERNAME}>"
        msg["To"] = recipient

        # Plain text fallback
        text_body = (
            f"Your Quant Engine verification code is: {otp}\n\n"
            f"This code expires in 5 minutes.\n\n"
            f"If you didn't request this, ignore this email."
        )
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(_build_otp_html(otp, recipient), "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, recipient, msg.as_string())

        logger.info(f"OTP email sent to {recipient}")
        return True

    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP authentication failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to send OTP email to {recipient}: {e}")
        return False
