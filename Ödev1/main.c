#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* --- Veri Tipi Tanýmlamalarý --- */
typedef enum {
    DTYPE_F32 = 0, // 32-bit Float (4 Byte)
    DTYPE_F16 = 1, // 16-bit Half-Float (2 Byte) - IEEE 754 half-precision
    DTYPE_I8  = 2  // 8-bit Integer (1 Byte) - Quantized
} TensorType;

/* --- Dinamik Tensör Yapýsý --- */
typedef struct {
    TensorType type;    // Veri tipi
    uint32_t length;    // Eleman sayýsý
    void* data;         // Veri bloðuna iþaretçi (RAM tasarrufu için dinamik)
    
    // Quantization parametreleri (Sadece I8 için kullanýlýr)
    float scale;
    int8_t zero_point;
} TinyTensor;

/* --- Fonksiyon Prototipleri --- */
TinyTensor* create_tensor(uint32_t len, TensorType type);
void destroy_tensor(TinyTensor* t);
void set_element_f32(TinyTensor* t, uint32_t index, float value);
float get_element_f32(TinyTensor* t, uint32_t index);

/*  Implementasyon  */

// Tensör için bellek ayýrma
TinyTensor* create_tensor(uint32_t len, TensorType type) {
    TinyTensor* t = (TinyTensor*)malloc(sizeof(TinyTensor));
    if (!t) return NULL;

    t->length = len;
    t->type = type;
    t->scale = 1.0f;      // Varsayýlan scale
    t->zero_point = 0;    // Varsayýlan offset

    // Veri tipi boyutuna göre bellek ayýr
    size_t element_size;
    switch (type) {
        case DTYPE_F32: element_size = sizeof(float); break;
        case DTYPE_F16: element_size = sizeof(uint16_t); break;
        case DTYPE_I8:  element_size = sizeof(int8_t); break;
        default: element_size = 0;
    }

    t->data = calloc(len, element_size);
    return t;
}

// Belirli bir indekse deðer atama (Otomatik dönüþüm ile)
void set_element_f32(TinyTensor* t, uint32_t index, float value) {
    if (index >= t->length) return;

    if (t->type == DTYPE_F32) {
        ((float*)t->data)[index] = value;
    } 
    else if (t->type == DTYPE_I8) {
        // Simple Linear Quantization: q = (v / scale) + zero_point
        int8_t q_val = (int8_t)((value / t->scale) + t->zero_point);
        ((int8_t*)t->data)[index] = q_val;
    }
    else if (t->type == DTYPE_F16) {
        // Gerçek projelerde burada float32 -> float16 dönüþüm algoritmasý olur
        // Örnek amaçlý uint16 cast yapýlmýþtýr
        ((uint16_t*)t->data)[index] = (uint16_t)value; 
    }
}

// Deðeri okuma (De-quantization dahil)
float get_element_f32(TinyTensor* t, uint32_t index) {
    if (index >= t->length) return 0.0f;

    if (t->type == DTYPE_F32) {
        return ((float*)t->data)[index];
    } 
    else if (t->type == DTYPE_I8) {
        // De-quantization: v = (q - zero_point) * scale
        int8_t q_val = ((int8_t*)t->data)[index];
        return (float)(q_val - t->zero_point) * t->scale;
    }
    return 0.0f;
}

void destroy_tensor(TinyTensor* t) {
    if (t) {
        free(t->data);
        free(t);
    }
}

/* --- Örnek Kullaným --- */
int main() {
    // 1. Bir Quantized (INT8) tensör oluþturalým (RAM tasarrufu)
    TinyTensor* my_layer = create_tensor(10, DTYPE_I8);
    my_layer->scale = 0.05f;      // Örn: Hassasiyet
    my_layer->zero_point = 0;

    // 2. Float veri yazalým (Ýçeride int8'e sýkýþtýrýlacak)
    set_element_f32(my_layer, 0, 12.5f);
    
    // 3. Veriyi geri okuyalým (Float olarak dönecek)
    printf("Index 0 (Quantized): %d\n", ((int8_t*)my_layer->data)[0]);
    printf("Index 0 (De-quantized): %.2f\n", get_element_f32(my_layer, 0));

    destroy_tensor(my_layer);
    return 0;
}
