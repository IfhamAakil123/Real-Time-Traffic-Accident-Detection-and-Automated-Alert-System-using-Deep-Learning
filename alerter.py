"""
alerter.py
----------
Sends alerts via multiple channels when an accident is detected:
  • Email (Gmail SMTP)
  • Windows desktop notification (plyer)
  • WhatsApp message (Twilio)
  • SMS message (Twilio)
"""

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.image     import MIMEImage
from pathlib              import Path
from datetime             import datetime


# ── Configuration helpers ─────────────────────────────────────────────────────
def _get_email_config():
    return {
        'sender'  : os.environ.get('ALERT_EMAIL_SENDER',   ''),
        'password': os.environ.get('ALERT_EMAIL_PASSWORD', ''),
        'receiver': os.environ.get('ALERT_EMAIL_RECEIVER', ''),
    }


def _get_twilio_config():
    return {
        'account_sid'   : os.environ.get('TWILIO_ACCOUNT_SID',    ''),
        'auth_token'    : os.environ.get('TWILIO_AUTH_TOKEN',      ''),
        'phone_from'    : os.environ.get('TWILIO_PHONE_NUMBER',    ''),
        'whatsapp_from' : os.environ.get('TWILIO_WHATSAPP_FROM',   ''),
    }


# ── Evidence helper ──────────────────────────────────────────────────────────
def _get_evidence_path(result: dict) -> str | None:
    """
    Return the absolute path to the best evidence screenshot.
    Prefers the highest-confidence frame; falls back to first flagged frame.
    """
    base = Path(__file__).parent / 'static'

    # Prefer best_evidence_frame (highest confidence)
    best = result.get('best_evidence_frame')
    if best:
        p = base / best
        if p.exists():
            return str(p)

    # Fallback: first flagged image
    if result.get('flagged_images'):
        p = base / result['flagged_images'][0]
        if p.exists():
            return str(p)

    return None


# ── Desktop notification ─────────────────────────────────────────────────────
def _desktop_notify(confidence: float, filename: str) -> bool:
    """Show a Windows desktop toast notification."""
    try:
        from plyer import notification
        notification.notify(
            title='Accident Detected!',
            message=f'File: {filename}\nConfidence: {confidence:.1%}\nCheck the dashboard for details.',
            app_name='Accident Detection System',
            timeout=10,
        )
        print('[Alert] Desktop notification sent.')
        return True
    except Exception as e:
        print(f'[Alert] Desktop notification failed: {e}')
        return False


# ── Email ────────────────────────────────────────────────────────────────────
def _send_email(result: dict, screenshot_path: str | None = None,
                alert_email: str = '') -> bool:
    """Send an HTML email with detection details and an optional screenshot."""
    cfg = _get_email_config()

    # Form-provided email takes priority over environment variable
    receiver = alert_email or cfg['receiver']

    if not cfg['sender'] or not cfg['password'] or not receiver:
        print('[Alert] Email credentials not configured — skipping email alert.')
        print('        Visit /settings in the web UI to enter your Gmail credentials.')
        return False

    try:
        ts      = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        subject = f'[ALERT] Road Accident Detected — {ts}'

        first_frame = result.get('first_accident_frame', 'N/A')
        duration    = result.get('duration_seconds')
        dur_str     = f'{duration:.1f}s' if duration else 'N/A'

        html_body = f"""
        <html><body style="font-family: Arial, sans-serif; color: #222;">
          <div style="background:#c0392b;color:#fff;padding:16px 24px;border-radius:6px 6px 0 0;">
            <h2 style="margin:0;">Road Accident Detected</h2>
            <p style="margin:4px 0 0;">{ts}</p>
          </div>
          <div style="background:#f9f9f9;padding:20px 24px;border:1px solid #ddd;border-radius:0 0 6px 6px;">
            <table style="border-collapse:collapse;width:100%;">
              <tr><td style="padding:6px 12px;font-weight:bold;width:180px;">File</td>
                  <td style="padding:6px 12px;">{result.get('original_filename', result.get('filename','N/A'))}</td></tr>
              <tr style="background:#fff;"><td style="padding:6px 12px;font-weight:bold;">Confidence</td>
                  <td style="padding:6px 12px;color:#c0392b;font-weight:bold;">
                      {result.get('max_confidence', 0):.1%}</td></tr>
              <tr><td style="padding:6px 12px;font-weight:bold;">First accident frame</td>
                  <td style="padding:6px 12px;">{first_frame}</td></tr>
              <tr style="background:#fff;"><td style="padding:6px 12px;font-weight:bold;">Total frames analysed</td>
                  <td style="padding:6px 12px;">{result.get('total_frames', 'N/A')}</td></tr>
              <tr><td style="padding:6px 12px;font-weight:bold;">Flagged frames</td>
                  <td style="padding:6px 12px;">{result.get('flagged_frames', 'N/A')}</td></tr>
              <tr style="background:#fff;"><td style="padding:6px 12px;font-weight:bold;">Video duration</td>
                  <td style="padding:6px 12px;">{dur_str}</td></tr>
            </table>
            {"<p style='margin-top:16px;'>Screenshot of the highest-confidence accident frame is attached below.</p>" if screenshot_path else ""}
          </div>
        </body></html>
        """

        msg = MIMEMultipart('related')
        msg['Subject'] = subject
        msg['From']    = cfg['sender']
        msg['To']      = receiver
        msg.attach(MIMEText(html_body, 'html'))

        if screenshot_path and Path(screenshot_path).exists():
            with open(screenshot_path, 'rb') as f:
                img = MIMEImage(f.read(), name=Path(screenshot_path).name)
                msg.attach(img)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(cfg['sender'], cfg['password'])
            smtp.sendmail(cfg['sender'], receiver, msg.as_string())

        print(f'[Alert] Email sent to {receiver}')
        return True

    except Exception as e:
        print(f'[Alert] Email failed: {e}')
        return False


# ── WhatsApp (Twilio) ────────────────────────────────────────────────────────
def _send_whatsapp(result: dict, alert_phone: str = '') -> bool:
    """Send a WhatsApp message via Twilio with accident details."""
    cfg = _get_twilio_config()

    if not cfg['account_sid'] or not cfg['auth_token'] or not cfg['whatsapp_from']:
        print('[Alert] Twilio WhatsApp not configured — skipping.')
        return False

    if not alert_phone:
        print('[Alert] No WhatsApp recipient phone — skipping.')
        return False

    try:
        from twilio.rest import Client

        client = Client(cfg['account_sid'], cfg['auth_token'])

        confidence = result.get('max_confidence', 0)
        filename   = result.get('original_filename', result.get('filename', 'N/A'))
        flagged    = result.get('flagged_frames', 0)

        body = (
            f"🚨 *ACCIDENT DETECTED*\n\n"
            f"📁 File: {filename}\n"
            f"📊 Confidence: {confidence:.1%}\n"
            f"🎞️ Flagged frames: {flagged}\n"
            f"⏱️ Duration: {result.get('duration_seconds', 'N/A')}s\n\n"
            f"Check the dashboard for full details."
        )

        # Format the recipient for WhatsApp
        to_number = alert_phone.strip()
        if not to_number.startswith('whatsapp:'):
            to_number = f'whatsapp:{to_number}'

        whatsapp_from = cfg['whatsapp_from']
        if not whatsapp_from.startswith('whatsapp:'):
            whatsapp_from = f'whatsapp:{whatsapp_from}'

        message = client.messages.create(
            body=body,
            from_=whatsapp_from,
            to=to_number,
        )

        print(f'[Alert] WhatsApp sent to {to_number} (SID: {message.sid})')
        return True

    except Exception as e:
        print(f'[Alert] WhatsApp failed: {e}')
        return False


# ── SMS (Twilio) ─────────────────────────────────────────────────────────────
def _send_sms(result: dict, alert_phone: str = '') -> bool:
    """Send an SMS message via Twilio with accident details."""
    cfg = _get_twilio_config()

    if not cfg['account_sid'] or not cfg['auth_token'] or not cfg['phone_from']:
        print('[Alert] Twilio SMS not configured — skipping.')
        return False

    if not alert_phone:
        print('[Alert] No SMS recipient phone — skipping.')
        return False

    try:
        from twilio.rest import Client

        client = Client(cfg['account_sid'], cfg['auth_token'])

        confidence = result.get('max_confidence', 0)
        filename   = result.get('original_filename', result.get('filename', 'N/A'))

        body = (
            f"ACCIDENT DETECTED\n"
            f"File: {filename}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Flagged frames: {result.get('flagged_frames', 0)}\n"
            f"Check dashboard for details."
        )

        # Strip any whatsapp: prefix for SMS
        to_number = alert_phone.strip()
        if to_number.startswith('whatsapp:'):
            to_number = to_number.replace('whatsapp:', '')

        message = client.messages.create(
            body=body,
            from_=cfg['phone_from'],
            to=to_number,
        )

        print(f'[Alert] SMS sent to {to_number} (SID: {message.sid})')
        return True

    except Exception as e:
        print(f'[Alert] SMS failed: {e}')
        return False


# ── Main alert dispatcher ────────────────────────────────────────────────────
def send_alert(result: dict, alert_email: str = '',
               alert_phone: str = '') -> dict:
    """
    Call after detection when result['accident_detected'] is True.
    Sends alerts via all configured channels.

    Returns a dict of channel statuses: {email, desktop, whatsapp, sms}
    """
    status = {'email': False, 'desktop': False, 'whatsapp': False, 'sms': False}

    if not result.get('accident_detected'):
        return status

    filename   = result.get('original_filename', result.get('filename', 'unknown'))
    confidence = result.get('max_confidence', 0.0)

    # Desktop notification
    status['desktop'] = _desktop_notify(confidence, filename)

    # Get best evidence screenshot (highest confidence frame)
    screenshot = _get_evidence_path(result)

    # Email alert
    status['email'] = _send_email(result, screenshot_path=screenshot,
                                  alert_email=alert_email)

    # WhatsApp alert
    status['whatsapp'] = _send_whatsapp(result, alert_phone=alert_phone)

    # SMS alert
    status['sms'] = _send_sms(result, alert_phone=alert_phone)

    return status
