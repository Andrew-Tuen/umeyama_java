import Jama.SingularValueDecomposition;
import Jama.Matrix;


public double[][] umeyama(double[][] src, double[][] dst, boolean estimate_scale)
{
        int num = src.length;
        int dim = src[0].length;

        //compute mean of src and dst
        double[] src_mean = new double[dim];
        double[] dst_mean = new double[dim];
        for (int i = 0; i < dim; i++)
        {
                src_mean[i] = 0;
                dst_mean[i] = 0;
        }
        for (int i = 0; i < num; i++)
        {
                for (int j = 0; j < dim; j++)
                {
                        src_mean[j] += src[i][j];
                        dst_mean[j] += dst[i][j];
                }
        }
        for (int i = 0; i < dim; i++)
        {
                src_mean[i] /= num;
                dst_mean[i] /= num;
        }

        // Subtract mean from src and dst.
        double[][] src_demean_f = new double[num][dim];
        double[][] dst_demean_f = new double[num][dim];
        for (int i = 0; i < num; i++)
        {
                for (int j = 0; j < dim; j++)
                {
                        src_demean_f[i][j] = src[i][j] - src_mean[j];
                        dst_demean_f[i][j] = dst[i][j] - dst_mean[j];
                }
        }

        // Eq. (38).
        Matrix src_demean = new Matrix(src_demean_f);
        Matrix dst_demean = new Matrix(dst_demean_f);
        Matrix A = dst_demean.transpose().times(src_demean).times(1.0/(double)num);

        //Eq. (39).
        double[][] d = new double[dim][dim];
        for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                {
                        d[i][j] = 0; 
                        if (i == j) d[i][j] = 1;
                }
        if (A.det() < 0)
                d[dim - 1][dim - 1] = -1;
        double[][] T = new double[dim+1][dim+1];

        Matrix S = A.svd().getS();
        Matrix V = A.svd().getV();
        Matrix U = A.svd().getU();

        double[][] temp_S = new double[1][dim];
        for (int i = 0; i < dim; i++)
        {
                temp_S[0][i] = S.get(i,i);
        }
        S = new Matrix(temp_S);

        V = V.transpose();

        //Eq. (40) and (43).
        int rank = A.rank();
        if (rank == 0)
        {
                for (int i = 0; i <= dim; i++)
                        for (int j = 0; j <= dim; j++)
                                T[i][j] = -1;
                return T;
        }
        else if (rank == dim -1)
        {
                if (U.det()*V.det() > 0)
                {
                        Matrix tempt = U.times(V);
                        for (int i = 0; i < dim; i++)
                        {
                                for (int j = 0; j < dim; j++)
                                {
                                        T[i][j] = tempt.get(i,j);
                                }
                        }
                }
                else
                {
                        double s = d[dim-1][dim-1];
                        d[dim-1][dim-1] = -1;
                        Matrix tempd = new Matrix(d);
                        Matrix tempt = U.times(tempd.times(V));
                        for (int i = 0; i < dim; i++)
                        {
                                for (int j = 0; j < dim; j++)
                                {
                                        T[i][j] = tempt.get(i,j);
                                }
                        }
                        d[dim-1][dim-1] = s;
                }
        }
        else
        {
                Matrix tempd = new Matrix(d);
                Matrix tempt = U.times(tempd.times(V));
                for (int i = 0; i < dim; i++)
                {
                        for (int j = 0; j < dim; j++)
                        {
                                T[i][j] = tempt.get(i,j);
                        }
                }
        }

        // Eq. (41) and (42).
        double scale = 1.0;
        if (estimate_scale)
        {
                double[] temp_mean = new double[dim];
                double[] temp_var = new double[dim];
                double[][] temp_d = new double[dim][1];
                for (int i = 0; i < dim; i++)
                {
                        temp_mean[i] = 0;
                        temp_var[i] = 0;
                        temp_d[i][0] = d[i][i];
                }
                for (int i = 0; i < num; i++)
                {
                        for (int j = 0; j < dim; j++)
                        {
                                temp_mean[j] += src_demean_f[i][j];
                        }
                }
                for (int i = 0; i < dim; i++)
                {
                        temp_mean[i] /= num;
                }
                for (int i = 0; i < num; i++)
                {
                        for (int j = 0; j < dim; j++)
                        {
                                temp_var[j] += (src_demean_f[i][j]-temp_mean[j])*(src_demean_f[i][j]-temp_mean[j]);
                        }
                }
                for (int i = 0; i < dim; i++)
                {
                        temp_var[i] /= num;
                }
                Matrix td = new Matrix(temp_d);
                double t2 = S.times(td).get(0,0);
                double t1 = 0;
                for (int i = 0; i < dim; i++)
                {
                        t1 += temp_var[i]; 
                }
                scale = 1.0 / t1 * t2;
        }
        double[][] temp_dst_mean_f = new double[dim][1];
        double[][] temp_src_mean_f = new double[dim][1];
        for (int i = 0; i < dim; i++)
        {
                temp_dst_mean_f[i][0] = dst_mean[i];
                temp_src_mean_f[i][0] = src_mean[i];
        }

        Matrix temp_dst_mean = new Matrix(temp_dst_mean_f);
        Matrix temp_src_mean = new Matrix(temp_src_mean_f);
        double[][] temp_T_f = new double[dim][dim];
        for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                        temp_T_f[i][j] = T[i][j];
        Matrix temp_T = new Matrix(temp_T_f);
        Matrix TT = temp_dst_mean.minus(temp_T.times(temp_src_mean).times(scale));
        for (int i = 0; i < dim; i++)
        {
                T[i][dim] = TT.get(i,0);
                for (int j = 0; j < dim; j++)
                {
                        T[i][j] *= scale;
                }
        }
        return T;
}