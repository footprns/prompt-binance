# Vanguard Android MDM Knowledge Base

Welcome to the **Vanguard Mobile Device Management (MDM)** knowledge base for **Android devices**.
This repository provides guidance, best practices, troubleshooting steps, and code samples for integrating and managing Android devices using Vanguard MDM.

---

## 📁 Table of Contents

- [📖 Overview](#overview)
- [📱 Supported Features](#supported-features)
- [🔧 Setup & Enrollment](#setup--enrollment)
- [🛡️ Security Policies](#security-policies)
- [⚙️ Configuration Profiles](#configuration-profiles)
- [📤 App Deployment](#app-deployment)
- [🔍 Troubleshooting](#troubleshooting)
- [🧪 API Usage & Samples](#api-usage--samples)
- [📚 References](#references)

---

## 📖 Overview

Vanguard MDM is a platform for securing, monitoring, and managing Android devices in enterprise environments. It enables IT administrators to:

- Enroll and provision devices.
- Enforce security configurations.
- Remotely deploy apps.
- Monitor compliance status.

---

## 📱 Supported Features

| Feature                      | Supported |
|-----------------------------|-----------|
| Device Enrollment           | ✅        |
| Work Profile Provisioning   | ✅        |
| Full Device Management      | ✅        |
| App Blacklisting/Whitelisting | ✅     |
| Remote Wipe / Lock          | ✅        |
| Location Tracking           | ✅        |
| Custom Configuration Payloads | ✅     |

---

## 🔧 Setup & Enrollment

### Prerequisites

- Vanguard MDM Admin Account
- Android 8.0 or higher
- Google Play Services enabled

### Enrollment Methods

1. **QR Code Enrollment** (zero-touch like experience)
2. **AFW (Android for Work)**
3. **Device Owner via ADB**

#### Example QR Enrollment

```
{  
  "android.app.extra.PROVISIONING_DEVICE_ADMIN_COMPONENT_NAME": "com.vanguard.mdm/.DeviceAdminReceiver",
  "android.app.extra.PROVISIONING_DEVICE_ADMIN_PACKAGE_DOWNLOAD_LOCATION": "https://mdm.vanguard.com/apk/latest.apk"
}
```

---

## 🛡️ Security Policies

- Password Requirements
- Disable Camera, Bluetooth
- Restrict Factory Reset
- Block USB Debugging

### Example JSON Policy

```json
{
  "passwordRequired": true,
  "minimumPasswordLength": 6,
  "disableCamera": true,
  "blockUsbDebugging": true
}
```

---

## ⚙️ Configuration Profiles

- Wi-Fi Settings
- VPN Configuration
- Email Account Provisioning

### Example: Wi-Fi Profile
```json
{
  "wifiSSID": "CorpNet",
  "securityType": "WPA2",
  "password": "corp-password"
}
```

---

## 📤 App Deployment

- Managed Google Play Integration
- APK Upload
- Silent Install (via Device Owner)

### Example API Call
```bash
curl -X POST https://api.vanguard.com/devices/{id}/apps \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{ "packageName": "com.example.myapp", "version": "1.2.3" }'
```

---

## 🔍 Troubleshooting

| Issue                        | Resolution                            |
|-----------------------------|----------------------------------------|
| Enrollment Failed           | Check device connectivity and QR config |
| App Not Installing          | Ensure app is approved and pushed       |
| Policy Not Applying         | Ensure device is in compliance state    |

---

## 🧪 API Usage & Samples

- Get Device Info
```bash
curl https://api.vanguard.com/devices/{device_id} -H "Authorization: Bearer <TOKEN>"
```

- Push Policy
```bash
curl -X POST https://api.vanguard.com/devices/{device_id}/policy \
  -H "Authorization: Bearer <TOKEN>" \
  -d @policy.json
```

---

## 📚 References

- [Android Enterprise](https://developer.android.com/work)
- [Google Play EMM API](https://developers.google.com/android/management)
- [Vanguard MDM Documentation](https://docs.vanguard.com/mdm)

---

> Maintained by the Vanguard MDM Security Team. For support, open a GitHub Issue or email `support@vanguard.com`.
