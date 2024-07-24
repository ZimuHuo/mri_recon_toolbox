from tqdm import tqdm
from scipy import interpolate
import numpy as np
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def blip_reversed_correction_line_integral(contracted, extended, oversampling = 50, overgrid = 10):
    [ny,nx] = contracted.shape
    recon = np.zeros([ny, nx])
    for x in tqdm(range(nx)):
        sig1 = np.cumsum(np.abs(contracted[:,x]))
        sig2 = np.cumsum(np.abs(extended[:,x]))
        avg = (sig1+sig2)/2
        maxval = (max(sig1)+ max(sig2))/2
        minval = (min(sig1)+min(sig2)) / 2
        f1 =  interpolate.interp1d(np.linspace(0,1,ny), sig1, kind='cubic', bounds_error=False, fill_value=0)
        f2 =  interpolate.interp1d(np.linspace(0,1,ny), sig2, kind='cubic', bounds_error=False, fill_value=0)
        y1s = f1(np.linspace(0, 1, ny*overgrid))
        y2s = f2(np.linspace(0, 1, ny*overgrid))
        list_idx = []
        list_val = []
        for val in (np.linspace(minval, maxval, ny * oversampling)):
            x1 = find_nearest(y1s, val) / overgrid
            x2 = find_nearest(y2s, val) / overgrid
            if (x1+x2)/2 not in list_idx:
                list_idx.append((x1+x2)/2)
                list_val.append(val)
        list_idx.append(ny)
        list_val.append(maxval)
        idx = np.asarray(list_idx)
        val = np.asarray(list_val)
        f = interpolate.interp1d(idx, val, kind='linear', bounds_error=False, fill_value=0)
        b = f(np.arange(ny))
        b[b>maxval] = 0
        for i in range(1,b.size):
            if b[i] < b[i-1]:
                b[i] = b[i-1]
        recon[:,x] = np.ediff1d(b, to_begin=b[0])
    return recon 

import SimpleITK as sitk
import sys
import os


def command_iteration(method, bspline_transform) :
    if method.GetOptimizerIteration() == 0:
        # The BSpline is resized before the first optimizer
        # iteration is completed per level. Print the transform object
        # to show the adapted BSpline transform.
        print(bspline_transform)


    print("{0:3} = {1:10.5f}".format(method.GetOptimizerIteration(),
                                     method.GetMetricValue()))



def command_multi_iteration(method) :
    # The sitkMultiResolutionIterationEvent occurs before the
    # resolution of the transform. This event is used here to print
    # the status of the optimizer from the previous registration level.
    if R.GetCurrentLevel() > 0:
        print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
        print(" Iteration: {0}".format(R.GetOptimizerIteration()))
        print(" Metric value: {0}".format(R.GetMetricValue()))

    print("--------- Resolution Changing ---------")


def blip_reversed_correction_bspline_registration(target, moving):
    target = np.abs(target).astype(np.float64)
    moving = np.abs(moving).astype(np.float64)
    fixed = sitk.GetImageFromArray(target)
    moving = sitk.GetImageFromArray(moving)

    transformDomainMeshSize=[2]*fixed.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed,
                                          transformDomainMeshSize )


    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsJointHistogramMutualInformation()

    R.SetOptimizerAsGradientDescentLineSearch(5.0,
                                              100,
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)

    R.SetInterpolator(sitk.sitkLinear)

    R.SetInitialTransformAsBSpline(tx,
                                   inPlace=True,
                                   scaleFactors=[1,2,5])
    R.SetShrinkFactorsPerLevel([4,2,1])
    R.SetSmoothingSigmasPerLevel([4,2,1])

    # R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R, tx) )
    # R.AddCommand( sitk.sitkMultiResolutionIterationEvent, lambda: command_multi_iteration(R) )

    outTx = R.Execute(fixed, moving)

    # print("-------")
    # print(tx)
    # print(outTx)
    # print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    # print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    # print(" Metric value: {0}".format(R.GetMetricValue()))

    # sitk.WriteTransform(outTx,  sys.argv[3])

    if ( not "SITK_NOSHOW" in os.environ ):

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed);
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(outTx)

        out = resampler.Execute(moving)
        # # simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
        # out = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
        # cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)
        # ref1 = sitk.GetArrayFromImage(simg2)
        # out = ref1.astype(np.float64)
        ref1 = sitk.GetArrayFromImage(out)
        out = ref1.astype(np.float64) 
    return out