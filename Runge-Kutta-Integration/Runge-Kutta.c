#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 2

void test(){
	printf("Hello World\n");
}

typedef double (*equacoes)();

//omega
double f0(double x, double y[N], double param[N]){
  return y[1];
}

//adicionar g/omega
double f1(double x, double y[N], double param[N]){
  return  -param[0]/param[1]*sin(y[0]);
}


void RK4(double x, double y[N], double h, double param[N]){
	double k1[N], k2[N], k3[N], k4[N], yp[N], ypp[N];
  equacoes f[] = {f0, f1};
  int i;

  for(i=0; i<N; i++){
    k1[i] = f[i](x,y, param);
    yp[i] = y[i] + h/2.*k1[i];
  }
  for(i=0; i<N; i++){
    k2[i]=f[i](x+h/2., yp, param);
    ypp[i] = y[i]+h/2.*k2[i];
  }
  for(i=0; i<N; i++){
    k3[i] = f[i](x+h/2., ypp, param);
    yp[i] = y[i]+h*k3[i];
  }
  for(i=0; i<N; i++){
    k4[i] = f[i](x+h, yp, param);
    y[i] += h/6.*(k1[i]+2*k2[i]+2*k3[i]+k4[i]);
  }
}

int main(void) {
	// y[N]={S, I, R} iniciais
	double a, b, h, x, y[N], param[N], theta_ini[4];
	int i, j;
	FILE *fp;
	
	a=0;
	b=20;
	h=0.01;

	//theta, v iniciais
	y[0] = 30*M_PI/180;
	y[1] = 0.;
	
	//g, l
	param[0] = 9.8;
	param[1] = 1;

	//valores para theta
	theta_ini[0] = 10*M_PI/180;
	theta_ini[1] = 45*M_PI/180;
	theta_ini[2] = 90*M_PI/180;
	theta_ini[3] = 170*M_PI/180;


	fp = fopen("rungekutta.csv", "w");

	//loop para as condições iniciais
	for(j=0; j<1; j++){
		y[0] = theta_ini[j];
		for(x=a;x<b;x+=h){
		  printf("%g \t", x);
		  fprintf(fp, "%g \t", x);
		  for(i=0; i<N; i++){
			printf(",%g \t", y[i]);
			fprintf(fp, ", %g \t", y[i]);
		  }
		  RK4(x,y,h, param);
		  puts("");
		  fprintf(fp, "\n");
		}
		fclose(fp);
		puts("\n\n");
	}
	


	
	return 0;
}

