

#include <Servo.h>
#include <ArduinoJson.h>

#define BAUD            9600

// ===== LCD (optional) =====
#define USE_I2C_LCD   1
#define LCD_ADDR      0x27
#define LCD_COLS      16
#define LCD_ROWS      2

#if USE_I2C_LCD
  #include <Wire.h>
  #include <LiquidCrystal_I2C.h>
  LiquidCrystal_I2C lcd(LCD_ADDR, LCD_COLS, LCD_ROWS);
#endif

// ===== Servo pins =====
#define SERVO_X_PIN   9   // pan
#define SERVO_Y_PIN   10  // tilt

// ===== Input & Mapping =====
#define IN_MIN_DEG    -90.0f
#define IN_MAX_DEG    +90.0f
#define X_CENTER      90.0f
#define Y_CENTER      90.0f
#define X_INVERT      0
#define Y_INVERT      0
#define X_MIN_OUT     10
#define X_MAX_OUT     170
#define Y_MIN_OUT     20
#define Y_MAX_OUT     160

// ===== Filtering =====
#define EMA_ALPHA     0.35f
#define RATE_LIMIT    6.0f   // deg/loop

// ===== Coupling Modes =====
#define COUPLE_INDEPENDENT   0
#define COUPLE_CROSS         1
#define COUPLE_LOCKSTEP      2
#define COUPLE_MODE          COUPLE_CROSS   // << เลือกโหมดที่นี่

// ค่ากำลังทิศหลักและครอส (เหมาะกับสเกล ~±17°)
#define GX    1.00f   // gain ของ X ไป SX
#define GY    1.00f   // gain ของ Y ไป SY
#define CXY   0.50f   // cross gain (Y→SX, X→SY)

// Deadband ใช้ตัดสินทิศทางแสดงผล
#define DIR_DEADBAND  0.5f

// ===== Packet buffer =====
static const size_t LINE_BUF = 256;
char lineBuf[LINE_BUF];
size_t lineLen = 0;
uint32_t lastRxMs = 0;
#define PACKET_TIMEOUT_MS 50

// ===== State =====
Servo servoX, servoY;
float xTarget = X_CENTER, yTarget = Y_CENTER;
float xSmooth = X_CENTER, ySmooth = Y_CENTER;
bool  tracking = false;

// คำสั่งล่าสุดจาก Python (หน่วย deg ภาพ)
float last_cmd_left = 0.0f;  // - = ซ้าย, + = ขวา
float last_cmd_up   = 0.0f;  // + = ขึ้น, - = ลง

// ===== Utils =====
float clampf(float v,float lo,float hi){ return v<lo?lo:(v>hi?hi:v); }
float ema(float prev,float tgt,float a){ return prev + a*(tgt-prev); }
float rateLim(float prev,float tgt,float md){ float d=tgt-prev; if(d>md)d=md; if(d<-md)d=-md; return prev+d; }

float mapInputToServo(float inDeg, float center, bool invert){
  if (invert) inDeg = -inDeg;
  float t = (inDeg - IN_MIN_DEG) / (IN_MAX_DEG - IN_MIN_DEG); // 0..1
  float out = t * 180.0f + (center - 90.0f);
  return out;
}

const char* dirLR(float left_deg){
  if (left_deg >  DIR_DEADBAND)  return "R";
  if (left_deg < -DIR_DEADBAND)  return "L";
  return "C";
}
const char* dirUD(float up_deg){
  if (up_deg >  DIR_DEADBAND)  return "U";
  if (up_deg < -DIR_DEADBAND)  return "D";
  return "C";
}

void lcdShowXYandDir(){
#if USE_I2C_LCD
  lcd.setCursor(0,0);
  char line0[21];
  snprintf(line0, sizeof(line0), "X:%-5.1f %-1s  Y:%-4.1f %-1s",
           last_cmd_left, dirLR(last_cmd_left), last_cmd_up, dirUD(last_cmd_up));
  for (int i=0;i<LCD_COLS;i++){
    char c = (i < (int)strlen(line0)) ? line0[i] : ' ';
    lcd.print(c);
  }
  lcd.setCursor(0,1);
  char line1[21];
  snprintf(line1, sizeof(line1), "SX:%-3d  SY:%-3d", (int)xSmooth, (int)ySmooth);
  for (int i=0;i<LCD_COLS;i++){
    char c = (i < (int)strlen(line1)) ? line1[i] : ' ';
    lcd.print(c);
  }
#endif
}

void sendAck(const char* status,const char* err=nullptr){
  StaticJsonDocument<160> doc;
  doc["ok"] = (strcmp(status,"ok")==0 || strcmp(status,"ready")==0);
  doc["status"] = status;
  doc["pan_x"] = (int)xSmooth;
  doc["tilt_y"]= (int)ySmooth;
  doc["tracking"] = tracking;
  doc["dir_x"] = dirLR(last_cmd_left);
  doc["dir_y"] = dirUD(last_cmd_up);
  if (err) doc["error"] = err;
  serializeJson(doc, Serial);
  Serial.println();
}

void applyMotion(){
  float xn = ema(xSmooth, xTarget, EMA_ALPHA);
  float yn = ema(ySmooth, yTarget, EMA_ALPHA);
  xn = rateLim(xSmooth, xn, RATE_LIMIT);
  yn = rateLim(ySmooth, yn, RATE_LIMIT);
  if ((int)xn != (int)xSmooth) servoX.write((int)xn);
  if ((int)yn != (int)ySmooth) servoY.write((int)yn);
  xSmooth = xn; ySmooth = yn;
}

void computeCoupledMix(float inX, float inY, float &outX, float &outY){
#if (COUPLE_MODE == COUPLE_INDEPENDENT)
  outX = inX;
  outY = inY;
#elif (COUPLE_MODE == COUPLE_CROSS)
  // SX = GX·X + CXY·Y,  SY = GY·Y + CXY·X
  outX = GX*inX + CXY*inY;
  outY = GY*inY + CXY*inX;
#elif (COUPLE_MODE == COUPLE_LOCKSTEP)
  // ให้ทั้งสองแกนมี "ขนาด" เท่ากัน = ค่าเฉลี่ยของ |X|,|Y| แต่ทิศตามสัญญาณเดิม
  float m = (fabs(inX) + fabs(inY)) * 0.5f;
  outX = (inX==0 ? 0 : copysignf(m, inX));
  outY = (inY==0 ? 0 : copysignf(m, inY));
#else
  outX = inX; outY = inY;
#endif
}

void handleJson(const char* s){
  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, s);
  if (err){ sendAck("bad_json", err.c_str()); return; }

  if (doc.containsKey("is_tracking")) tracking = doc["is_tracking"].as<bool>();

  bool got=false;
  if (doc.containsKey("left_deg")) { last_cmd_left = doc["left_deg"].as<float>(); got=true; }
  if (doc.containsKey("up_deg"))   { last_cmd_up   = doc["up_deg"].as<float>();   got=true; }

  // ---- Coupling ----
  float x_mix, y_mix;
  computeCoupledMix(last_cmd_left, last_cmd_up, x_mix, y_mix);

  // ---- Map → Clamp → Target ----
  float xo = mapInputToServo(x_mix, X_CENTER, X_INVERT);
  float yo = mapInputToServo(y_mix, Y_CENTER, Y_INVERT);
  xTarget  = clampf(xo, X_MIN_OUT, X_MAX_OUT);
  yTarget  = clampf(yo, Y_MIN_OUT, Y_MAX_OUT);

  sendAck(got ? "ok" : "json_no_ctrl");
}

void flushPacket(){
  if (lineLen==0) return;
  while(lineLen>0 && (lineBuf[lineLen-1]=='\n' || lineBuf[lineLen-1]=='\r')) lineLen--;
  lineBuf[lineLen]='\0';

  if (lineLen>0) handleJson(lineBuf); else sendAck("empty");

  // อัปเดตจอทุกครั้งที่มีแพ็กเก็ต
  lcdShowXYandDir();

  // บอกเหตุการณ์ด้วยไฟ LED
  digitalWrite(LED_BUILTIN, HIGH); delay(40); digitalWrite(LED_BUILTIN, LOW);

  lineLen=0;
}

void setup(){
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.begin(BAUD);
  while(!Serial){;}

#if USE_I2C_LCD
  Wire.begin();
  lcd.init(); lcd.backlight(); lcd.clear();
  lcd.setCursor(0,0); lcd.print("XY Coupled Ready");
  lcd.setCursor(0,1); lcd.print("@"); lcd.print(BAUD); lcd.print(" bps");
#endif

  servoX.attach(SERVO_X_PIN);
  servoY.attach(SERVO_Y_PIN);
  servoX.write((int)xSmooth);
  servoY.write((int)ySmooth);

  sendAck("ready");
  lastRxMs = millis();
}

void loop(){
  while(Serial.available()){
    char c = (char)Serial.read();
    if (lineLen < LINE_BUF-1) lineBuf[lineLen++] = c;
    lastRxMs = millis();
    if (c=='\n') flushPacket();
  }
  if (lineLen>0 && (millis()-lastRxMs)>PACKET_TIMEOUT_MS) flushPacket();

  applyMotion();

#if USE_I2C_LCD
  static uint32_t tUI=0; uint32_t now=millis();
  if (now - tUI > 200) { tUI = now; lcdShowXYandDir(); }
#endif
}
