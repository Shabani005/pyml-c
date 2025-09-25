#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define K_NUM 5
#define normalize(arr) cml_normalizef((arr), sizeof(arr) / sizeof((arr)[0]))
#define LEARNING_R 0.1f
#define OUTPUT 10


#define cml_knn_fit(data, labels, train) \
    cml_knn_fit_impl(                   \
        sizeof(data) / sizeof((data)[0]), \
        sizeof((data)[0]) / sizeof((data)[0][0]), \
        data,                           \
        labels,                         \
        train                           \
    )
#define cml_fit_linear(x, y) cml_fit_linear_impl((x), (y), sizeof(x)/sizeof((x)[0]))

// KNN Struct
typedef struct{
  float *features;
  int label;
} point;

typedef struct{
  point* points;
  size_t size;
  size_t dim;
} dataset;

typedef struct{
  dataset train;
  dataset test;
} dataset_split;

typedef struct{
  point point;
  float dist;
} tosort;


// LINREG Struct
typedef struct{
  float* x;
  float* y;
  size_t size;
} sample;


// GENERAL FUNCTIONS
float cml_random_float();


// KNN FUNCTIONS
int cml_compare_tosort(const void *a, const void *b); 
float cml_distance_vec(point a, point b, size_t dim);
void cml_shuffle_dataset(point *arr, size_t n);
int cml_knn_predict(dataset train, point test_pt, int num_classes);
void cml_knn_train(dataset_split full_data); //#undef K_NUM -> #define K_NUM <custom_value>
dataset_split cml_knn_fit_impl(size_t n, size_t dim, float data[n][dim], int *labels, float train);

// LINREG FUNCTIONS 
float* cml_normalizef(float array[], size_t size); // Use Macro Instead
sample cml_fit_linear_impl(float *x_arr, float* y_arr, size_t size);
float cml_train_linear(sample data, int epochs);

#ifdef CML_IMPLEMENTATION
int cml_compare_tosort(const void *a, const void *b) {
    float d1 = ((tosort *)a)->dist;
    float d2 = ((tosort *)b)->dist;
    return (d1 > d2) - (d1 < d2);
}

float cml_distance_vec(point a, point b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a.features[i] - b.features[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

void cml_shuffle_dataset(point *arr, size_t n) {
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        point tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

int cml_knn_predict(dataset train, point test_pt, int num_classes){
  int *label_count = calloc(num_classes, sizeof(int));
  tosort* sarray = malloc(sizeof(tosort) * train.size);

  for (int i=0; i < train.size; ++i){
    sarray[i].point = train.points[i];
    sarray[i].dist = cml_distance_vec(test_pt, train.points[i], train.dim);
  }
  qsort(sarray, train.size, sizeof(tosort), cml_compare_tosort);
  
  for (size_t i = 0; i < K_NUM; ++i){
    label_count[sarray[i].point.label]++;
  }

  int predicted_label = -1;
  int max_count = -1;

  for (int lbl = 0; lbl < num_classes; ++lbl){
    if (label_count[lbl] > max_count){
    max_count = label_count[lbl];
    predicted_label = lbl;
    }
  }
  return predicted_label;
}


dataset_split cml_knn_fit_impl(size_t n, size_t dim, float data[n][dim], int *labels, float train){
  printf("TOTALSIZE: %zu, DIM: %zu\n", n, dim);

  point *points = malloc(n * sizeof(point));
  if (!points) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(1);
  }

  for (size_t i = 0; i < n; i++) {
      points[i].features = malloc(dim * sizeof(float));
      for (size_t j = 0; j < dim; j++) {
          points[i].features[j] = data[i][j];
      }
      points[i].label = labels[i];
  }

  cml_shuffle_dataset(points, n);
  size_t train_size = (size_t)(n * train);

  dataset train_ds = {points, train_size, dim};
  dataset test_ds = {points + train_size, n - train_size, dim};
  
  dataset_split full = {train_ds, test_ds};
  return full;
}


void cml_knn_train(dataset_split full_data){
  dataset test = full_data.test;
  dataset train = full_data.train;

  int correct = 0;
  size_t dim = test.dim;  

  for (size_t i = 0; i < test.size; ++i) {
      int pred = cml_knn_predict(train, test.points[i], 3);
      printf("Test point (");
      for (size_t j = 0; j < dim; j++) {
          printf("%.1f%s", test.points[i].features[j],
                 (j < dim - 1) ? ", " : "");
      }
      printf(") true=%d pred=%d\n", test.points[i].label, pred);
      if (pred == test.points[i].label) correct++;
  }

  printf("\nAccuracy: %.2f%%\n", (100.0 * correct) / test.size);
}

float* cml_normalizef(float array[], size_t size){
  float highest = array[0];
  float lowest = array[0];

  for( size_t i=0; i < size; ++i){
   if (array[i] > highest){
      highest = array[i];
    }
    if (array[i] < lowest){
      lowest = array[i];
    }
  }
    for (size_t i = 0; i < size; ++i) {
        array[i] = (array[i] - lowest) / (highest - lowest);
    }
        return array;
  }

float cml_random_float(){
    return (float) rand() / (float) RAND_MAX;
}


sample cml_fit_linear_impl(float *x_arr, float *y_arr, size_t size){
  sample fit_custom = {
    .x = cml_normalizef(x_arr, size),
    .y = cml_normalizef(y_arr, size),
    .size = size
  };
  return fit_custom;
}

float cml_train_linear(sample data, int epochs){
  srand(time(0)); 
  float weight = cml_random_float();
  int EPOCHS = epochs;
  for (int epoch = 0; epoch < EPOCHS; ++epoch){
    float total_loss = 0.0f;
    float grad = 0.0f;

    for (int i=0; i < data.size; ++i){
      float x = data.x[i]; 
      float y = data.y[i];
      float y_hat = x * weight;
      float error = y_hat - y; 
      total_loss += error*error;
      grad += error*x;
    }
    total_loss /= data.size;
    grad /= data.size;
    
    weight -= LEARNING_R*grad;
    if (epoch % (EPOCHS/OUTPUT) == 0){
      printf("Epoch: %d Loss: %.6e Weight: %f\n", epoch, total_loss, weight);
    }
    
  }
    printf("\nTrained weight: %f\n", weight);
  
  for (int i = 0; i < data.size; ++i) {
      float x = data.x[i];
      float y = data.y[i];
      float y_pred = weight * x;
      printf("x = %f, y = %f, y_pred = %f\n", x, y, y_pred);
  }
  return weight;
}

#endif

#ifdef CML_STRIP_PREFIX
#define compare_tosort    cml_compare_tosort
#define distance_vec       cml_distance_vec
#define shuffle_dataset    cml_shuffle_dataset
#define knn_predict        cml_knn_predict
#define random_float       cml_random_float
#define fit_linear         cml_fit_linear
#define train_linear       cml_train_linear
#define normzlizef         cml_normalizef
#define knn_fit            cml_knn_fit
#define knn_train          cml_knn_train
#define knn_predict        cml_knn_predict
#define fit_linear_impl    cml_fit_linear_impl
#endif
