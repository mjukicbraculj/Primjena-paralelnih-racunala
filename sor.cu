#include  "cuda_wrapper.h"
#include <stdio.h>

double g(double x, double y)
{
    //double rez=sin(x);
    //printf("rez koji vracam za %d %d je %lg, ...\n", x, y, rez);
   // double rez=-sin(x)-sin(y);
    //double rez=x*x+y*y;
    double rez=sin(x)+y*y;
    return rez;
}
void provjeri(double *polje, int size, int Ni){
	double h = (double)1/Ni;
	double max_raz=0, pamtim_vrijednost1, pamtim_vrijednost2, tmp, raz;
	int i, j, pamtim_index, pamtim_i, pamtim_j;
	
	for(i=0; i<Ni-1; ++i)
        for(j=0; j<Ni-1; ++j)     ///x je j, y je i
        {
            tmp=sin((j+1)*h)+(i+1)*h*(i+1)*h;
           // tmp=-sin((j+1)*h)-sin((i+1)*h);
            //if(i*Ni+j==0)printf("%lg\n", tmp);
            raz=fabs(tmp-polje[i*(Ni-1)+j]);
           // printf("%d je index, %lg je funkcijska %lg je iz datoteke\n", i*Ni+j, tmp, polje[i*Ni+j]);
            if(raz>max_raz)
            {
                //printf("tu sam %d\n", i*(Ni-1)+j);
                max_raz=raz;
				pamtim_i = i +1;
				pamtim_j = j + 1;
                pamtim_index=i*(Ni-1)+j;
                pamtim_vrijednost1=tmp;
                pamtim_vrijednost2=polje[i*(Ni-1)+j];
            }
        }
    printf("maximalno odstupanje je %.16lg, na indexu %d(%d, %d), %.16lg je vrijednost funkcije, %.16lg je vrijednost aproksimsacije\n", max_raz, pamtim_index, pamtim_i, pamtim_j, pamtim_vrijednost1, pamtim_vrijednost2);
	
}
//za ovo isto napisati kernel
void generiraj_f(double *f, int size, int Ni){
	double h = (double)1/Ni;
	printf("%lg je h\n", h);
	for(int k = 0; k < size; ++k){
		int x = (k%(Ni-1))+1;
		int y = (k/(Ni-1))+1;
		f[k]=-sin(x*h)+2;
        f[k]=f[k]*h*h;
		if(x==1&&y==1)f[k]-=(g((x-1)*h,y*h)+g(x*h,(y-1)*h));      ///sredjivanje rubnih uvjeta
        else if(x==Ni-1&&y==1)f[k]-=(g(x*h,(y-1)*h)+g((x+1)*h,y*h));
        else if(x==1&&y==Ni-1)f[k]-=(g((x-1)*h,y*h)+g(x*h,(y+1)*h));
        else if(x==Ni-1&&y==Ni-1)f[k]-=(g((x+1)*h,y*h)+g(x*h,(y+1)*h));
        else if(x==1)f[k]-=g((x-1)*h,y*h);
        else if(x==Ni-1)f[k]-=g((x+1)*h,y*h);
        else if(y==1)f[k]-=g(x*h,(y-1)*h);
        else if(y==Ni-1)f[k]-=g(x*h,(y+1)*h);
        else continue;
	}
}

void ispisi(double *polje, int size){
	for(int i = 0; i < size; ++i)
		printf("%lg, ", polje[i]);
	printf("\n");
}

__device__ int visekratink(int broj, int N){
	while(broj > 0)
		broj -= N;
	if(broj == 0)
		return 1;
	return 0;

}


__device__ double f_cpu(double x, double y){
	return -sin(x) + 2;
}
__device__ double g_cpu(double x, double y){
	return sin(x) + y*y;
}


__global__ void generirajF(double *f, int Ni){
	int moj_index_u_retku = blockIdx.x * blockDim.x + threadIdx.x;
	double h = (double)1/(Ni+1);
	double rez;
	if(moj_index_u_retku < Ni){
		rez = f_cpu((moj_index_u_retku+1)*h, (blockIdx.y + 1)*h);
		rez *= h * h;
		if(moj_index_u_retku == 0)
			rez -= g_cpu(0, (blockIdx.y+1) * h);
		if(moj_index_u_retku == Ni-1)
			rez -= g_cpu(1, (blockIdx.y+1) * h);
		if(blockIdx.y == 0)
			rez -= g_cpu((moj_index_u_retku+1)*h, 0);
		if(blockIdx.y == Ni -1)
			rez -= g_cpu((moj_index_u_retku + 1)*h, 1); 
		//if(blockIdx.y == 0)
			//printf("%d je moj index i dobio sam %lg i spremam na mjesto %d\n", moj_index_u_retku, rez, Ni * blockIdx.y + moj_index_u_retku);
		f[Ni*blockIdx.y + moj_index_u_retku] = rez;
	}
}

__global__ void black_red(double *f, double *in_data, double *out_data, int black, int size, int Ni){
	//nadjemo stvarni index, pomonozimo ga s 2 i dodamo black
	//ako je black == 0 onda gledamo samo parni, inace gledamo samo parne
	int moj_index = blockIdx.x * blockDim.x + threadIdx.x;
	int moj_stvarni_index = moj_index * 2 + black;
	
	
	if(moj_stvarni_index < size){
		double sum = 0;
		if(moj_stvarni_index - Ni + 1 >= 0){
			sum += in_data[moj_stvarni_index-Ni+1];
			//printf("%d index i dodajem mu sudjeda %d\n", moj_stvarni_index, moj_stvarni_index - Ni + 1);
		}
		if(moj_stvarni_index + Ni -1 <= size-1){
			sum += in_data[moj_stvarni_index+Ni-1];
			//printf("%d index i dodajem mu sudjeda %d\n", moj_stvarni_index, moj_stvarni_index + Ni - 1);
		}
		if(moj_stvarni_index - 1 >= 0 && !visekratink(moj_stvarni_index, Ni - 1)){
			sum += in_data[moj_stvarni_index-1];
			//printf("%d index i dodajem mu sudjeda %d\n", moj_stvarni_index, moj_stvarni_index - 1);
		}
		if(moj_stvarni_index + 1 <= size-1 && !visekratink(moj_stvarni_index+1, Ni - 1)){
			sum += in_data[moj_stvarni_index+1];
			//printf("%d index i dodajem mu sudjeda %d\n", moj_stvarni_index, moj_stvarni_index + 1);			
		}
		
		
		sum -= f[moj_stvarni_index];
		out_data[moj_stvarni_index] = sum/4;	
		//printf("index: %d, suma: %lg, f: %lg, out_data: %lg\n",moj_stvarni_index, sum, f[moj_stvarni_index], out_data[moj_stvarni_index]);
		//printf("index prije i index nakon %lg, %lg\n", out_data[moj_stvarni_index-1], out_data[moj_stvarni_index+1]);
	}
	
}

__global__ void greska(double* f, double *in_data, double *out_data, int size, int Ni){
	int moj_stvarni_index = blockIdx.x * blockDim.x + threadIdx.x;
	if(moj_stvarni_index < size){
		double sum = 0;
		if(moj_stvarni_index - Ni + 1 >= 0){
			sum += in_data[moj_stvarni_index-Ni+1];
			//printf("%d index i dodajem mu sudjeda %d\n", moj_stvarni_index, moj_stvarni_index - Ni + 1);
		}
		if(moj_stvarni_index + Ni -1 <= size-1){
			sum += in_data[moj_stvarni_index+Ni-1];
			//printf("%d index i dodajem mu sudjeda %d\n", moj_stvarni_index, moj_stvarni_index + Ni - 1);
		}
		if(moj_stvarni_index - 1 >= 0 && !visekratink(moj_stvarni_index, Ni - 1)){
			sum += in_data[moj_stvarni_index-1];
			//printf("%d index i dodajem mu sudjeda %d\n", moj_stvarni_index, moj_stvarni_index - 1);
		}
		if(moj_stvarni_index + 1 <= size-1 && !visekratink(moj_stvarni_index+1, Ni - 1)){
			sum += in_data[moj_stvarni_index+1];
			//printf("%d index i dodajem mu sudjeda %d\n", moj_stvarni_index, moj_stvarni_index + 1);			
		}
		
		
		sum -= 4 * in_data[moj_stvarni_index];
		out_data[moj_stvarni_index] = f[moj_stvarni_index] - sum;
		//printf("racunanje greske: %d index, %lg greska, suma: %lg, f: %lg\n", moj_stvarni_index, out_data[moj_stvarni_index], sum, f[moj_stvarni_index]);
	}
}
int main(int argc, char** argv){
	
	//s komandne linije učitamo Ni (na koliko intervala dijelimo segment)
	//i učitamo epsilon
	
	if(argc != 3){
		printf("ucitati: broj intervala i epsilon");
		exit(EXIT_FAILURE);
	}
	
	int Ni = atoi(argv[1]);
	double epsilon = atof(argv[2]);
	
	double gpu_nrm;
	
	//izračunamo velicinu polja u i f
	int size = (Ni + 1) * (Ni + 1) - 2 * (Ni + 1) - 2 * (Ni - 1);
	
	//grid i block
	dim3 grid;
	dim3 block;
	dim3 grid_error;
	dim3 block_error;
	
	double		cpu_time = 0.0;
	double		gpu_time = 0.0;
	
	
	double *hst_f = NULL;
	double *hst_u = NULL;
	double *hst_eps = NULL;
	host_alloc(hst_f, double, size * sizeof(double));
	host_alloc(hst_u, double, size * sizeof(double));
	host_alloc(hst_eps, double, size * sizeof(double));
	
	cublasHandle_t	handle;
	cublas_exec(cublasCreate(&handle));
	cublas_exec(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
	
	
	
	//ispisi(hst_f, size);
	//u je na pocetku nula
	
	double *dev_f = NULL;
	double *dev_u1 = NULL;
	double *dev_u2 = NULL;
	double *dev_eps = NULL;
	cuda_exec(cudaMalloc(&dev_f, size * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_u1, size * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_u2, size * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_eps, size * sizeof(double)));
	cuda_exec(cudaMemset(dev_u1, 0, size * sizeof(double)));
	cuda_exec(cudaMemset(dev_u2, 0, size * sizeof(double)));
	
	//generiraj_f(hst_f, size, Ni);
	block.x = 256;
	block.y = 1;
	grid.x = (Ni + block.x -1)/block.x;
	grid.y = Ni;
	
	generirajF<<<grid, block>>>(dev_f, Ni-1);
	//prekopiramo  hst_f u dev_f
	//cuda_exec(cudaMemcpy(dev_f, hst_f, size * sizeof(double), cudaMemcpyHostToDevice));
	cuda_exec(cudaMemcpy(hst_f, dev_f, size * sizeof(double), cudaMemcpyDeviceToHost));
	
	/*printf("ispisujemo dio f\n");
	for(int i = 0; i < 10; ++i)
		printf("%lg, ", hst_f[i]);
	printf("\n");*/
	
	
	//imamo u i f, sada moramo napraviti grid
	block.x = 256;
	block.y = 1;
	
	grid.x = (((Ni+1)*(Ni+1))/2 + block.x -1)/block.x;
	grid.y = 1;
	
	
	block_error.x = 256;
	block_error.y = 1;
	
	grid_error.x = ((Ni+1)*(Ni+1) + block_error.x -1)/block_error.x;
	grid_error.y = 1;
	
	printf("duljina bloka je %d, a duljina grida je %d\n", block.x, grid.x);
	printf("duljina bloka je %d, a duljina grida je %d\n", block_error.x, grid_error.x);
	//ideja je napraviti kernel koji radi jednu iteraciju i onda podatke vraća na host jer
	//je to jedan od načina kako možemo sinkronizirati sve dretve
	
	int crni = 0;
	printf("duljina arrayja je %d\n", size);
	double *error;
	int iter = 0;
	
	gpu_time -= timer();
	
	while(1){
		if(crni){
			//pozivamo kernel za parne indexe
			black_red<<<grid, block>>>(dev_f, dev_u1, dev_u2, crni, size, Ni);
			cuda_exec(cudaMemcpy(hst_u, dev_u2, size * sizeof(double), cudaMemcpyDeviceToHost));
			cuda_exec(cudaMemcpy(dev_u1, hst_u, size * sizeof(double), cudaMemcpyHostToDevice));
			greska<<<grid_error, block_error>>>(dev_f, dev_u1, dev_eps,  size, Ni);
			cublas_exec(cublasDnrm2(handle, size, dev_eps, 1, &gpu_nrm));
			//printf("%lg je norma greske\n", gpu_nrm);
			if(gpu_nrm < epsilon)
				break;
			//ispisi(hst_u, size);
		}
		else{
			black_red<<<grid, block>>>(dev_f, dev_u1, dev_u2, crni, size, Ni);
			cuda_exec(cudaMemcpy(hst_u, dev_u2, size * sizeof(double), cudaMemcpyDeviceToHost));
			cuda_exec(cudaMemcpy(dev_u1, hst_u, size * sizeof(double), cudaMemcpyHostToDevice));
			//ispisi(hst_u, size);
		}
		crni = 1 - crni;
		++iter;
		//printf("-----------------------------------\n");
	}
	cuda_exec(cudaDeviceSynchronize());
	gpu_time += timer();
	printf("%lg je norma greske\n", gpu_nrm);
	
	//ispisi(hst_u, size);
	provjeri(hst_u, size, Ni);

	printf("GPU execution time: %dms, broj iteracija %d\n", int(1000*gpu_time), iter);
	
	
	return 0;
	
}