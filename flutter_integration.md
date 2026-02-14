# Flutter Integration Guide

This guide explains how to connect your Flutter frontend to the ConversaVoice backend.

## 1. Network Configuration

The tricky part is connecting your phone/emulator to `localhost`.

### A. Android Emulator
Use the special alias IP:
```dart
static const String baseUrl = 'http://10.0.2.2:8000';
```

### B. iOS Simulator
Use localhost directly:
```dart
static const String baseUrl = 'http://localhost:8000';
```

### C. Physical Device (Same Wi-Fi)
1. Find your PC's IP address (e.g., `ipconfig` on Windows -> `192.168.1.5`)
2. Use that IP:
```dart
static const String baseUrl = 'http://192.168.1.5:8000';
```
*Note: Make sure to allow port 8000 through your Windows Firewall.*

---

## 2. Dependencies
Add `http` to your `pubspec.yaml`:
```yaml
dependencies:
  http: ^1.2.0
  uuid: ^4.3.3  # For session management
```

---

## 3. Dart API Implementation (`api_service.dart`)

Copy this code into your Flutter project:

```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:uuid/uuid.dart';

class ApiService {
  // CHANGE THIS to match your setup (see Network Configuration above)
  static const String baseUrl = 'http://10.0.2.2:8000'; 
  
  String? _sessionId;
  final Uuid _uuid = const Uuid();

  // Create or retrieve session ID
  String get sessionId {
    _sessionId ??= _uuid.v4();
    return _sessionId!;
  }

  // 1. Health Check
  Future<bool> checkHealth() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/api/health'));
      return response.statusCode == 200;
    } catch (e) {
      print('Health check failed: $e');
      return false;
    }
  }

  // 2. Chat with Text
  Future<Map<String, dynamic>> chat(String text) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/chat'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'text': text,
        'session_id': sessionId,
      }),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to chat: ${response.body}');
    }
  }

  // 3. Upload Audio for Transcription
  Future<String> transcribe(File audioFile) async {
    var request = http.MultipartRequest(
      'POST', 
      Uri.parse('$baseUrl/api/transcribe'),
    );
    
    request.fields['session_id'] = sessionId;
    request.files.add(await http.MultipartFile.fromPath(
      'audio', 
      audioFile.path,
    ));

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['text'];
    } else {
      throw Exception('Transcription failed: ${response.body}');
    }
  }

  // 4. Synthesize Speech (TTS)
  // Returns the URL of the generated audio
  Future<String> synthesize(String text, {String? style}) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/synthesize'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'text': text,
        'style': style,
      }),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      // Construct full URL
      return '$baseUrl${data["audio_url"]}';
    } else {
      throw Exception('TTS failed: ${response.body}');
    }
  }
}
```

## 4. Example Usage

```dart
final api = ApiService();

// Check if backend is running
bool isOnline = await api.checkHealth();

// Send text
try {
  final result = await api.chat("Hello from Flutter!");
  print("Response: ${result['response']}");
  
  // Convert response to speech
  final audioUrl = await api.synthesize(result['response']);
  // Use a player like `audioplayers` to play audioUrl
} catch (e) {
  print("Error: $e");
}
```

---

## 5. Full API Reference

Here are all the available endpoints you can use in your app:

### A. General
*   **Health Check**: `GET /api/health`
    *   Returns: `{"status": "healthy", "services": {...}}`
*   **Create Session**: `POST /api/session`
    *   Returns: `{"session_id": "uuid...", "created_at": "..."}`
    *   *Tip: Call this when app starts to get a unique session ID.*

### B. Speech-to-Text (STT)
*   **Transcribe Audio**: `POST /api/transcribe`
    *   **Body**: Multipart Form Data
        *   `audio`: The audio file (wav/mp3)
        *   `session_id`: (Optional) The session ID
    *   **Response**: `{"text": "Hello world", "session_id": "..."}`

### C. Chat (LLM)
*   **Send Text**: `POST /api/chat`
    *   **Body (JSON)**:
        ```json
        {
          "text": "Hello",
          "session_id": "..."
        }
        ```
    *   **Response**:
        ```json
        {
          "response": "Hi there! How can I help?",
          "style": "friendly",
          "session_id": "..."
        }
        ```

### D. Text-to-Speech (TTS)
*   **Generate Audio**: `POST /api/synthesize`
    *   **Body (JSON)**:
        ```json
        {
          "text": "Hi there!",
          "style": "friendly"
        }
        ```
    *   **Response**: `{"audio_url": "/api/audio/filename.wav"}`

*   **Download Audio**: `GET /api/audio/{filename}`
    *   Use the full URL (e.g., `http://192.168.x.x:8000/api/audio/filename.wav`) to play the audio in Flutter.

