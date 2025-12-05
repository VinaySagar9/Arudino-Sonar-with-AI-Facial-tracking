#include <Servo.h>
#include <LiquidCrystal.h>

static const uint8_t PIN_SERVO = 9;  // servo signal pin
static const uint8_t PIN_TRIG  = 8;  // HC-SR04 trig pin
static const uint8_t PIN_ECHO  = 7;  // HC-SR04 echo pin

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);  // rs,en,d4,d5,d6,d7

static const int SERVO_MIN = 0;  // min sweep/follow angle
static const int SERVO_MAX = 150;  // max sweep/follow angle

static const int SWEEP_STEP_DEG = 1;  // sweep increment
static const uint16_t SWEEP_STEP_MS = 25;  // sweep timing

static const uint16_t FOLLOW_TICK_MS = 15;  // follow timing
static const int FOLLOW_MAX_STEP = 3;  // max follow step per tick

static const uint16_t REPORT_MS = 25;  // serial/LCD report rate

static const unsigned long PULSE_TIMEOUT_US = 25000;  // pulseIn timeout
static const int DIST_MIN_CM = 2;  // sensor min valid
static const int DIST_MAX_CM = 300;  // sensor max valid

enum Mode : uint8_t { MODE_SWEEP, MODE_FOLLOW };  // run modes

Servo radarServo;  // servo object

Mode mode = MODE_SWEEP;  // default mode
bool paused = false;  // stop motion toggle

int currentAngle = 0;  // current servo angle
int sweepAngle = 0;  // sweep position
int sweepDir = +1;  // sweep direction

int targetAngle = 90;  // follow target

unsigned long lastStepMs = 0;  // step timer
unsigned long lastReportMs = 0;  // report timer
unsigned long lastLCDMs = 0;  // lcd timer

static char lineBuf[64];  // serial line buffer
static uint8_t lineLen = 0;  // buffer length

static inline int clampInt(int v, int lo, int hi) {  // clamp helper
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static bool equalsIgnoreCase(const char* a, const char* b) {  // case-insensitive compare
  while (*a && *b) {
    char ca = *a, cb = *b;
    if (ca >= 'a' && ca <= 'z') ca -= 32;
    if (cb >= 'a' && cb <= 'z') cb -= 32;
    if (ca != cb) return false;
    a++; b++;
  }
  return (*a == '\0' && *b == '\0');
}

static bool startsWithIgnoreCase(const char* s, const char* prefix) {  // case-insensitive prefix check
  while (*prefix) {
    char cs = *s, cp = *prefix;
    if (!cs) return false;
    if (cs >= 'a' && cs <= 'z') cs -= 32;
    if (cp >= 'a' && cp <= 'z') cp -= 32;
    if (cs != cp) return false;
    s++; prefix++;
  }
  return true;
}

static void handleCommand(const char* line) {  // parse one command line
  if (!line || !*line) return;

  if (equalsIgnoreCase(line, "STOP")) { paused = true; return; }  // freeze motion
  if (equalsIgnoreCase(line, "GO"))   { paused = false; return; }  // resume motion

  if (equalsIgnoreCase(line, "SWEEP"))  { mode = MODE_SWEEP;  paused = false; return; }  // set sweep mode
  if (equalsIgnoreCase(line, "FOLLOW")) { mode = MODE_FOLLOW; paused = false; return; }  // set follow mode

  if (startsWithIgnoreCase(line, "TARGET:")) {  // TARGET:<angle>
    const char* p = line + 7;
    while (*p == ' ' || *p == '\t') p++;
    int ang = atoi(p);
    targetAngle = clampInt(ang, SERVO_MIN, SERVO_MAX);  // bound target
    mode = MODE_FOLLOW;  // switch to follow
    paused = false;  // ensure running
    return;
  }
}

static void pollSerialLines() {  // non-blocking serial line reader
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') continue;  // ignore CR

    if (c == '\n') {
      lineBuf[lineLen] = '\0';  // terminate string
      handleCommand(lineBuf);  // dispatch command
      lineLen = 0;  // reset buffer
    } else {
      if (lineLen < sizeof(lineBuf) - 1) lineBuf[lineLen++] = c;  // append char
      else lineLen = 0; // overflow -> reset
    }
  }
}

static long readRawDistCm() {  // one HC-SR04 reading
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);

  unsigned long duration = pulseIn(PIN_ECHO, HIGH, PULSE_TIMEOUT_US);  // echo pulse
  if (duration == 0) return -1;  // timeout

  long cm = (long)(duration * 0.0343f * 0.5f);  // us -> cm
  return cm;
}

static long getStableDistCm() {  // median-of-3 filter
  long a = readRawDistCm();
  long b = readRawDistCm();
  long c = readRawDistCm();

  long arr[3] = {a, b, c};
  for (int i = 0; i < 2; i++) {
    for (int j = i + 1; j < 3; j++) {
      if (arr[j] < arr[i]) { long t = arr[i]; arr[i] = arr[j]; arr[j] = t; }  // sort
    }
  }

  long m = arr[1];  // median
  if (m < DIST_MIN_CM || m > DIST_MAX_CM) return -1;  // invalid range
  return m;
}

static void reportAngleDistance(int angle, long distCm) {  // Serial: angle,distance
  Serial.print(angle);
  Serial.print(",");
  Serial.println(distCm);
}

static void lcdUpdate(const char* modeStr, int angle, long distCm) {  // update LCD w/ rate limit
  unsigned long now = millis();
  if (now - lastLCDMs < 150) return;  // throttle LCD
  lastLCDMs = now;

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(modeStr);
  lcd.print(" A:");
  lcd.print(angle);

  lcd.setCursor(0, 1);
  lcd.print("D:");
  if (distCm < 0) lcd.print("--");  // no reading
  else lcd.print((int)distCm);
  lcd.print("cm");
}

static void writeServoAngle(int deg) {  // bounded servo write
  deg = clampInt(deg, SERVO_MIN, SERVO_MAX);  // clamp range
  radarServo.write(deg);  // move servo
  currentAngle = deg;  // track state
}

static void stepTowardTarget() {  // follow stepper
  int diff = targetAngle - currentAngle;
  if (diff == 0) return;  // already there

  int step = diff;
  if (step >  FOLLOW_MAX_STEP) step =  FOLLOW_MAX_STEP;  // cap step
  if (step < -FOLLOW_MAX_STEP) step = -FOLLOW_MAX_STEP;  // cap step

  writeServoAngle(currentAngle + step);  // take one step
}

void setup() {
  Serial.begin(9600);  // serial link

  lcd.begin(16, 2);  // 16x2 LCD
  lcd.clear();
  lcd.print("Radar Online");  // boot banner
  delay(800);
  lcd.clear();

  pinMode(PIN_TRIG, OUTPUT);  // trig output
  pinMode(PIN_ECHO, INPUT);  // echo input

  radarServo.attach(PIN_SERVO);  // attach servo pin
  writeServoAngle(currentAngle);  // initial position
}

void loop() {
  pollSerialLines();  // handle incoming commands

  unsigned long now = millis();  // current time

  if (paused) {
    if (now - lastReportMs >= REPORT_MS) {
      lastReportMs = now;
      long d = getStableDistCm();  // distance reading
      reportAngleDistance(currentAngle, d);  // stream to PC
      lcdUpdate((mode == MODE_FOLLOW) ? "FOLLOW" : "SWEEP", currentAngle, d);  // LCD status
    }
    return;
  }

  if (mode == MODE_SWEEP) {
    if (now - lastStepMs >= SWEEP_STEP_MS) {
      lastStepMs = now;

      sweepAngle += sweepDir * SWEEP_STEP_DEG;  // advance sweep
      if (sweepAngle >= SERVO_MAX) { sweepAngle = SERVO_MAX; sweepDir = -1; }  // bounce high
      if (sweepAngle <= SERVO_MIN) { sweepAngle = SERVO_MIN; sweepDir = +1; }  // bounce low

      writeServoAngle(sweepAngle);  // move servo

      long d = getStableDistCm();  // distance reading
      reportAngleDistance(currentAngle, d);  // stream to PC
      lcdUpdate("SWEEP", currentAngle, d);  // LCD status
    }
    return;
  }

  if (now - lastStepMs >= FOLLOW_TICK_MS) {
    lastStepMs = now;
    stepTowardTarget();  // follow target angle
  }

  if (now - lastReportMs >= REPORT_MS) {
    lastReportMs = now;
    long d = getStableDistCm();  // distance reading
    reportAngleDistance(currentAngle, d);  // stream to PC
    lcdUpdate("FOLLOW", currentAngle, d);  // LCD status
  }
}
