using System.Runtime.InteropServices;

public static class libMachineLearningWrapper
{
    [DllImport("libMachineLearning")]
    public static extern int my_add(int x, int y);

    [DllImport("libMachineLearning")]
    public static extern int my_mul(int x, int y);
}