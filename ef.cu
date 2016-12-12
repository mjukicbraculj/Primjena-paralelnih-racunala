
#include <iostream>
#include "cimg/CImg.h"
#include <string>
#include <sstream>
#include  "cuda_wrapper.h"
#include "cublas_v2.h"
#include "magma.h"
#include "magma_lapack.h"
using namespace cimg_library;

std::string toString(int broj){
  std::ostringstream ss;
  ss << broj;
  return ss.str();
}


void provjeri_sumu_stupaca(float *matrica,  float *suma, int n, int m){
  for(int i = 0; i < n; ++i){
    float sum = 0;
    for(int j = 0; j < m; ++j){
      sum += matrica[j*n+i];
    }
    if(sum != suma[i]){
      std::cout << "greska u provjeri sume stupaca na indexu: " << i <<  std::endl;
      return;
    }
  }
  std::cout << "suma je tocno izracunata!" << std::endl;
}

void provjeri_nakon_oduzimanja(float *matrica_stara, float *matrica_nova,  float *suma, int n, int m){
  for(int j = 0;  j < m; ++j){
    for(int i = 0; i < n; ++i){
      if(abs((matrica_stara[j*n+i] - (suma[i]/m)) - matrica_nova[j*n+i]) >=0.001 ){
        std::cout << "greska nakon oduzimanja na indexu:(" << i << ", " << j;
        std::cout << ") " << matrica_stara[j*n+i] << ", " << matrica_nova[j*n+i];
        std::cout << ", " << (matrica_stara[j*n+i] - (suma[i]/m)) << std::endl;
        return;
      }
    }
  }
  std::cout << "tocno nakon oduzimanja" << std::endl;
}

int main(int argc, char **argv){

	magma_init();

	CImg<unsigned int> slika("orl_faces/s1/1.pgm");
	

	if(argc != 3){
		printf("krivi unos s komandne linije\n");
		exit(-1);
	}
	
	unsigned int width = slika.width(), height = slika.height();
	unsigned int n = width * height;    //broj redaka matrice
	unsigned int m = 400;                 //broj stupaca matrice = koliko slika imamo

	   //host memorija
	int size = 40 * 10 * width * height;
	  float *hst_matrica = NULL;
	  float *hst_matrica_tmp = NULL;
	  float *data = NULL;
	  float *hst_y = NULL;
	  float *hst_x = NULL;
	  float *hst_v = NULL;
	  float *hst_u = NULL;
	  int *hst_iwork = NULL;
	  float *hst_work = NULL;
	  host_alloc(hst_matrica, float, size * sizeof(float));
	  host_alloc(hst_matrica_tmp, float, size * sizeof(float));
	  host_alloc(data, float, n * sizeof(float));
	  host_alloc(hst_y, float, n * sizeof(float));
	  host_alloc(hst_x, float, n * sizeof(float));
	  host_alloc(hst_v, float, m * m * sizeof(float));
	  host_alloc(hst_u, float, n * n * sizeof(float));
	  host_alloc(hst_iwork, int, 8 * m * sizeof(int));
	  host_alloc(hst_work, float, sizeof(float));
	  
	  cuda_exec(cudaHostRegister(hst_matrica, size * sizeof(float), cudaHostRegisterDefault));


	  //alociramo memoriju na hostu
	  float *dev_x = NULL;
	  float *dev_y = NULL;
	  float *dev_g = NULL;
	  float * dev_matrica = NULL;
	  float *dev_u = NULL;
	  float *dev_v = NULL;
	  int *dev_iwork = NULL;
	  float *dev_work = NULL;
	  cuda_exec(cudaMalloc(&dev_x, n * sizeof(float)));
	  cuda_exec(cudaMalloc(&dev_y, n * sizeof(float)));
	  cuda_exec(cudaMalloc(&dev_g, n * m * sizeof(float)));
	  cuda_exec(cudaMalloc(&dev_matrica, n * m * sizeof(float)));
	  cuda_exec(cudaMalloc(&dev_u, n * n * sizeof(float)));
	  cuda_exec(cudaMalloc(&dev_v, m * m * sizeof(float)));
	  cuda_exec(cudaMalloc(&dev_iwork, 8 * m * sizeof(int)));
	  cuda_exec(cudaMalloc(&dev_work, sizeof(float)));
	  cuda_exec(cudaMemset(dev_y, 0, n * sizeof(float)));
	  cuda_exec(cudaMemset(dev_u, 0, n * n * sizeof(float)));

	  //cublas
	  cublasHandle_t  handle;
	  cublas_exec(cublasCreate(&handle));
	  cublas_exec(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));


	float alfa = 1;
	for(int i = 1; i < 41; ++i){
		for(int j=1; j < 11; ++j){
			std::string ime("orl_faces/s" + toString(i) + "/" + toString(j) + ".pgm");
			CImg<float> slika(ime.c_str());
			data = slika.data();
			int pomak = ((i-1) * 10 + (j-1)) * n;
			for(int m = 0; m < n; ++m){
				hst_matrica[pomak + m] = data[m];
			}

			cuda_exec(cudaMemcpy(dev_x, data, n * sizeof(float), cudaMemcpyHostToDevice));
			cublas_exec(cublasSaxpy(handle, n, &alfa, dev_x, 1, dev_y, 1 ));
		}
	}

	//provjera jesmo dobro izracunali
	cuda_exec(cudaMemcpy(hst_y, dev_y, n * sizeof(float), cudaMemcpyDeviceToHost));
	provjeri_sumu_stupaca(hst_matrica, hst_y, n, m);

	cuda_exec(cudaMemcpy(dev_matrica, hst_matrica, n * m * sizeof(float), cudaMemcpyHostToDevice));
	cuda_exec(cudaMemcpy(hst_matrica_tmp, hst_matrica, n * m * sizeof(float), cudaMemcpyHostToHost));

	//x ponisitmo na nule, dodamo mu 1/400 * y
	//sada u x imamo aritmeticku sredinu koju moramo oduzeti svim recima matrice
	float beta = -0.0025;
	//cuda_exec(cudaMemset(dev_x, 1, n * sizeof(float))); kako postaviit na 1 ??
	cuda_exec(cudaMemcpy(hst_y, dev_y, n * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < n; ++i)
		hst_x[i] = 1;
	cuda_exec(cudaMemcpy(dev_x, hst_x, n * sizeof(float), cudaMemcpyHostToDevice));

	//cuda_exec(cudaMemset(dev_matrica, 0, n * m * sizeof(float)));
	cublas_exec(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &beta, dev_y, n, dev_x, 1, &alfa, dev_matrica, n));
	cuda_exec(cudaMemcpy(hst_matrica, dev_matrica, n * m * sizeof(float), cudaMemcpyDeviceToHost));
	provjeri_nakon_oduzimanja(hst_matrica_tmp, hst_matrica, hst_y, n, m);
	
	
	//MAGMA
	int status;
	  
	magma_int_t ret = magma_sgesdd(MagmaAllVec, n, m, hst_matrica, n, hst_x, hst_u, n , hst_v,  m,  hst_work, -1, hst_iwork,  &status);
	  
	int lwork = hst_work[0];
	host_alloc(hst_work, float, lwork * sizeof(float));
	magma_int_t ret1 = magma_sgesdd(MagmaAllVec, n, m, hst_matrica, n, hst_x, hst_u, n , hst_v,  m,  hst_work, lwork, hst_iwork,  &status);
	  
	cuda_exec(cudaMemcpy(dev_u, hst_u, n * n * sizeof(float), cudaMemcpyHostToDevice));
	
	//racunamo U^T*A = G
	beta = 0;
	cublas_exec(cublasSgemm(handle,  CUBLAS_OP_T,  CUBLAS_OP_N, n, m, n, &alfa, dev_u, n, dev_matrica, n, &beta, dev_g, n));
	cuda_exec(cudaMemcpy(hst_matrica_tmp, dev_g, n * m *sizeof(float), cudaMemcpyDeviceToHost));
	
	//učitamo podatke trazene slike, oduzmemo aritmeticku sredinu
	CImg<float> trazena_slika(argv[1]);
	data = trazena_slika.data();
	alfa = -0.0025;
	cuda_exec(cudaMemcpy(hst_x, data, n*sizeof(float), cudaMemcpyHostToHost));
	cuda_exec(cudaMemcpy(dev_x, hst_x, n * sizeof(float), cudaMemcpyHostToDevice ));
	cublas_exec(cublasSaxpy(handle, n , &alfa, dev_y, 1, dev_x, 1));			//oduzimamo srednju vrijednost
	
	
	//U^t*x = x
	alfa = 1;
	beta = 0;
	cublas_exec(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, n, &alfa, dev_u, n,  dev_x, n, &beta, dev_y, n));	//u dev_y spremamo Ux
	
	//od G oduzimamo x
	beta = 1;
	alfa = -1;
	for(int i = 0; i < m; ++i)
		hst_x[i] = 1;
	cuda_exec(cudaMemcpy(dev_x, hst_x, m * sizeof(float), cudaMemcpyHostToDevice));
	cublas_exec(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &alfa, dev_y, n, dev_x, 1, &beta, dev_g, n));

	//trazimo stupac s najmanjim odstupanjem
	float eps;
	float min_eps;
	int min_index = 0;
	printf("prije fora\n");
	for(int i=0; i < m; ++i){	
		cublas_exec(cublasSnrm2(handle, n, dev_g + i * n, 1, &eps));
		if(eps < min_eps || i == 0){
			min_eps = eps;
			min_index = i;
		}
	}				
			
	
	printf("slika se nalazi u s%d,(pomoc %d)\n", (min_index/10)+1,  min_index%10+1);
	
	return 0;

}

/*g++ bonus1.cpp -o bonus1 -L/usr/X11R6/lib -lm -lpthread -lX11
*/
