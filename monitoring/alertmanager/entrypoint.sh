#!/bin/sh
set -eu

TEMPLATE=/etc/alertmanager/alertmanager.yml.template
FALLBACK=/etc/alertmanager/alertmanager.yml
OUT=/tmp/alertmanager.yml

if [ -n "${ALERT_EMAIL_TO:-}" ] && [ -n "${SMTP_FROM:-}" ] && [ -n "${SMTP_PASSWORD:-}" ]; then
  export SMTP_SMARTHOST="${SMTP_SMARTHOST:-smtp.gmail.com:587}"
  export SMTP_USERNAME="${SMTP_USERNAME:-$SMTP_FROM}"
  export SMTP_FROM SMTP_PASSWORD ALERT_EMAIL_TO
  envsubst '${SMTP_SMARTHOST} ${SMTP_FROM} ${SMTP_USERNAME} ${SMTP_PASSWORD} ${ALERT_EMAIL_TO}' \
    < "$TEMPLATE" > "$OUT"
else
  cp "$FALLBACK" "$OUT"
fi

EXTERNAL_URL="${ALERTMANAGER_EXTERNAL_URL:-http://127.0.0.1:9093}"
exec /bin/alertmanager \
  --config.file="$OUT" \
  --storage.path=/alertmanager \
  --web.external-url="$EXTERNAL_URL"
