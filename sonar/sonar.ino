#include <Servo.h>
#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2); // lcd pins
#define TRIG 8
#define ECHO 7

Servo radarServo;

bool paused = false;       // pause flag
int currentAngle = 0;      // store servo angle

void setup() {
  Serial.begin(9600);

  lcd.begin(16, 2);
  lcd.clear();
  lcd.print("Radar Online");
  delay(800);
  lcd.clear();

  radarServo.attach(9);    // servo pin

  pinMode(TRIG, OUTPUT);   // ultrasonic pins
  pinMode(ECHO, INPUT);
}

long readRawDist() {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);

  long duration = pulseIn(ECHO, HIGH, 25000); // timeout
  if (duration <= 0) return -1;

  return duration * 0.0343 / 2; // cm
}

long getStableDist() {
  long a = readRawDist();
  long b = readRawDist();
  long c = readRawDist();

  // median filter
  long arr[3] = {a, b, c};
  for (int i = 0; i < 2; i++) {
    for (int j = i+1; j < 3; j++) {
      if (arr[j] < arr[i]) {
        long t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
      }
    }
  }

  long m = arr[1];
  if (m < 2 || m > 300) return -1;
  return m;
}

void checkCommands() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();                           // remove \r, spaces

    if (cmd.equalsIgnoreCase("STOP")) {
      paused = true;
    }

    if (cmd.equalsIgnoreCase("GO")) {
      paused = false;
    }
  }
}

void printLCD(int angle, long dist) {
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Angle:");
  lcd.print(angle);

  lcd.setCursor(0,1);
  lcd.print("Dist:");
  lcd.print(dist);
  lcd.print("cm");
}

void loop() {

  checkCommands();
  if (paused) {
    radarServo.write(currentAngle); // hold still
    delay(20);
    return;
  }

  // forward sweep
  for (int a = 0; a <= 180; a++) {
    checkCommands();
    if (paused) return;

    currentAngle = a;
    radarServo.write(a);

    long d = getStableDist();

    Serial.print(a);
    Serial.print(",");
    Serial.println(d);

    printLCD(a, d);
    delay(25);
  }

  // backward sweep
  for (int a = 180; a >= 0; a--) {
    checkCommands();
    if (paused) return;

    currentAngle = a;
    radarServo.write(a);

    long d = getStableDist();

    Serial.print(a);
    Serial.print(",");
    Serial.println(d);

    printLCD(a, d);
    delay(25);
  }
}
