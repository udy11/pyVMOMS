# Last Updated: 2019-Jun-14

# VMOMS Code for finding Moment Solutions to Grad-Shafranov Equation

# Troubleshooting:
#   If "Singular Matrix" error appears, try increasing x0 in uvmoms()
#   If there are issues with np.linalg.solve, try np.linalg.lstsq (see
#     commented line there)

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import sys
from gbvpsolver import gbvpsolver

def thavg(f, dth):
    return scipy.integrate.simps(f, dx = dth) / np.pi

def geotrn1(trng, elip):
    ''' (float, float) --> float, float
        Calculates R2/R1 and E used in VMOMS from geometric
        ellipticity and triangularity
    '''
    ctc = (scipy.optimize.root(lambda x: 2 * x - 3 * x**3 - trng * (1 - 2 * x**2), 0.5 * trng).x)[0]
    ctc2 = ctc * ctc
    c2tc = 2.0 * ctc2 - 1
    return (trng - ctc) / (1.0 - c2tc), elip * np.sqrt(1.0 - ctc2) / (trng * ctc - c2tc)

def geotrn2(r1, r2, ee):
    ''' (float, float, float) --> float, float
        Calculates geometric ellipticity and triangularity from
        R1, R2 and E in VMOMS
    '''
    r2r1 = r2 / r1
    r2r12 = r2r1 * r2r1
    ctc = 4.0 * r2r1 / (1.0 + np.sqrt(1.0 + 32.0 * r2r12))
    ctc2 = ctc * ctc
    c2tc = 2.0 * ctc2 - 1
    stc = np.sqrt(1.0 - ctc2)
    s2tc = 2.0 * stc * ctc
    return r2r1 + ctc - r2r1 * c2tc, ee * (stc + r2r1 * s2tc)

def uvmoms(rmaj, tring, ellip, rhodt, presdt, curidt, p0in, ivp_method = 'RK45', rt_method = 'lm'):
    ''' (float, float, float, array, array, array, [str, str]) --> array, array
    
        Finds Moment Solutions to Grad-Shafranov Equation
        
        INPUT:
        rmaj = Major Radius (meters, float)
        tring = Geometric Triangularity desired at rhodt[-1] (unitless, float)
        ellip = Ellipticity desired at rhodt[-1] (unitless, float)
        rhodt = Grid on which pressure and current data are mentioned (meters, 1d float array)
        presdt = Pressure data on rhodt (pascal, 1d float array)
        curidt = Current data on rhodt (amperes, 1d float array)
        p0in = parameters used to initialize p0 (unitless, array)
               (rmaj/rmin - tring) + p0in[0] roughly gives R0 at rhodt[0]
               p0in[1] * rhodt[0]**2 roughly gives R2 at rhodt[0]
               p0in[2] roughly gives E at rhodt[0]
        ivp_method = method to be used by ODE IVP solver, check odesolver() for details (str, optional)
        rt_method = method to be used by root finder, check rootfinder() for details (str, optional)
        
        OUTPUT:
        ps = final grid on which solution was found, same as R1 (unitless, 1d float array)
        solv = final values of R0, R0x, R2, R2x, E, Ex on ps (unitless, 2d float array)
    '''
    mu0 = 4.0e-7 * np.pi
    lmb1 = 1.0e-4
    lmb2 = 1.0e-4
    ndt = len(rhodt)
    rho0 = np.max(rhodt)
    x0 = max(5.0e-3, min(3.0e-2, rhodt[1] / rho0))
    xdt = np.linspace(x0, 1, ndt)
    r2f, eef = geotrn1(tring, ellip)

    rga = rmaj / rho0
    pres0 = np.max(presdt)
    curi0 = np.max(curidt)
    pres = scipy.interpolate.CubicSpline(rhodt/rho0, presdt/pres0)
    curi = scipy.interpolate.CubicSpline(rhodt/rho0, curidt/curi0)

    p0 = np.array([rga - r2f + p0in[0], p0in[1], p0in[2]])

    def fy0(x0, p):
        return np.array([p[0] - lmb1 * x0**2,
                         -2.0 * lmb1 * x0,
                         p[1] * x0**2,
                         2.0 * p[1] * x0,
                         p[2] + lmb2 * x0**2,
                         2.0 * lmb2 * x0])

    def frt(x, y):
        return np.array([y[0, -1] - (rga - r2f),
                         y[2, -1] - r2f,
                         y[4, -1] - eef])

    # temporary variables to be used in fyp()
    th = np.linspace(0, np.pi, 129)
    dth = th[1] - th[0]
    cs1t = np.cos(th)
    cs2t = np.cos(2.0 * th)
    sn1t = np.sin(th)
    sn2t = np.sin(2.0 * th)
    pif0 = 4.0 * np.pi**2 * pres0 * rho0**2 / curi0**2 / mu0
    def fyp(x, y, p):

        rn = np.array([y[0], x, y[2]])
        rnp = np.array([y[1], 1.0, y[3]])
        ee = y[4]
        eep = y[5]
        prs = pres(x)
        prsp = pres(x,1)
        cri = curi(x)
        crip = curi(x,1)

        r = rn[0] - rn[1] * cs1t + rn[2] * cs2t
        rt = rn[1] * sn1t - 2.0 * rn[2] * sn2t
        rtt = rn[1] * cs1t - 4.0 * rn[2] * cs2t
        rp = rnp[0] - rnp[1] * cs1t + rnp[2] * cs2t
        rtp = rnp[1] * sn1t - 2.0 * rnp[2] * sn2t
        z = ee * (rn[1] * sn1t + rn[2] * sn2t)
        zt = ee * (rn[1] * cs1t + 2.0 * rn[2] * cs2t)
        ztt = -ee * (rn[1] * sn1t + 4.0 * rn[2] * sn2t)
        zp = eep * (rn[1] * sn1t + rn[2] * sn2t) + ee * (rnp[1] * sn1t + rnp[2] * sn2t)
        ztp = eep * (rn[1] * cs1t + 2.0 * rn[2] * cs2t) + ee * (rnp[1] * cs1t + 2.0 * rnp[2] * cs2t)
        sg = r * (rt * zp - rp * zt)
        g = sg ** 2
        gtt = rt ** 2 + zt ** 2
        gtp = rt * rp + zt * zp
        gs8 = gtt / g
        gs10 = gs8 * r * zt / sg
        gs12 = gs10 * cs2t
        gs21 = gs8 * r * rt * sn1t / sg

        gs22 = gs8 * r * rt * sn2t / sg
        gs5 = zt / (r ** 2 * sg)
        gs6 = (gtt * (-rtp * zp + rp * ztp) + gtp * (rtt * zp + rt * ztp - rtp * zt - rp * ztt)) * r / (sg ** 3)
        gs7 = (rt * rtp + zt * ztp - zp * ztt - rp * rtt) / g
        gs567 = gs5 + gs6 + gs7
        mr0 = r * zt
        mr2 = r * (ee * rt * sn2t - zt * cs2t)
        mee = r * rt * (rn[1] * sn1t + rn[2] * sn2t)

        asg = thavg(sg, dth)
        asgr2 = thavg(sg / (r ** 2), dth)
        asggs8 = thavg(gtt / sg, dth)
        asggs10 = thavg(sg * gs10, dth)
        asggs12 = thavg(sg * gs12, dth)
        asggs21 = thavg(sg * gs21, dth)
        asggs22 = thavg(sg * gs22, dth)
        cs1 = thavg(sg * gs567, dth) / asggs8

        tmpv0 = gs10 - gs8 * asggs10 / asggs8
        dmr010 = thavg(mr0 * tmpv0, dth)
        dmr210 = thavg(mr2 * tmpv0, dth)
        dmee10 = thavg(mee * tmpv0, dth)

        tmpv0 = gs12 - gs8 * asggs12 / asggs8
        dmr012 = thavg(mr0 * tmpv0, dth)
        dmr212 = thavg(mr2 * tmpv0, dth)
        dmee12 = thavg(mee * tmpv0, dth)

        tmpv0 = gs21 - gs8 * asggs21 / asggs8
        dmr021 = thavg(mr0 * tmpv0, dth)
        dmr221 = thavg(mr2 * tmpv0, dth)
        dmee21 = thavg(mee * tmpv0, dth)

        tmpv0 = gs22 - gs8 * asggs22 / asggs8
        dmr022 = thavg(mr0 * tmpv0, dth)
        dmr222 = thavg(mr2 * tmpv0, dth)
        dmee22 = thavg(mee * tmpv0, dth)

        tmpv0 = (1 - asg / (asgr2 * r ** 2))
        tmpv1 = -pif0 * prsp * (asggs8 ** 2) / cri ** 2
        b0p = tmpv1 * thavg(mr0 * tmpv0, dth)
        b2p = tmpv1 * thavg(mr2 * tmpv0, dth)
        bep = tmpv1 * thavg(mee * tmpv0, dth)

        tmpv0 = (gs8 - asggs8 / (asgr2 * r ** 2))
        tmpv1 = -(crip / cri)
        b0i = tmpv1 * thavg(mr0 * tmpv0, dth)
        b2i = tmpv1 * thavg(mr2 * tmpv0, dth)
        bei = tmpv1 * thavg(mee * tmpv0, dth)

        tmpv0 = -gs567 + cs1 * gs8
        b05678 = thavg(mr0 * tmpv0, dth)
        b25678 = thavg(mr2 * tmpv0, dth)
        be5678 = thavg(mee * tmpv0, dth)

        b0 = b0p + b0i + b05678 + 2.0 * eep * (dmr021 * rnp[1] + dmr022 * rnp[2])
        b2 = b2p + b2i + b25678 + 2.0 * eep * (dmr221 * rnp[1] + dmr222 * rnp[2])
        be = bep + bei + be5678 + 2.0 * eep * (dmee21 * rnp[1] + dmee22 * rnp[2])
        b = np.array([b0, b2, be])

        a00 = dmr010
        a20 = dmr210
        ae0 = dmee10
        a02 = dmr012 - ee * dmr022
        a22 = dmr212 - ee * dmr222
        ae2 = dmee12 - ee * dmee22
        a0e = -(rn[1] * dmr021 + rn[2] * dmr022)
        a2e = -(rn[1] * dmr221 + rn[2] * dmr222)
        aee = -(rn[1] * dmee21 + rn[2] * dmee22)
        a = np.array([[a00, a02, a0e], [a20, a22, a2e], [ae0, ae2, aee]])
        #yxx = np.linalg.lstsq(a, b, rcond=-1)[0]
        yxx = np.linalg.solve(a, b)

        return np.array([rnp[0], yxx[0], rnp[2], yxx[1], eep, yxx[2]])

    ps, solv = gbvpsolver(lambda x: xdt, fyp, fy0, frt, p0, ivp_method = ivp_method, rt_method = rt_method)
    return ps, solv

if __name__ == '__main__':

    presdt = 1.602176487e-16 * 1.0e20 * np.array([22.4992714, 22.4934483, 22.4818058, 22.4643478, 22.4410896, 22.4120426, 22.3772259, 22.3366604, 22.2903690, 22.2383804, 22.1807289, 22.1174450, 22.0485706, 21.9741459, 21.8942165, 21.8088322, 21.7180405, 21.6219006, 21.5204678, 21.4138050, 21.3019772, 21.1850491, 21.0630932, 20.9361839, 20.8043938, 20.6678066, 20.5265007, 20.3805637, 20.2300797, 20.0751419, 19.9158401, 19.7522717, 19.5845318, 19.4127216, 19.2369442, 19.0573006, 18.8739014, 18.6868515, 18.4962616, 18.3022442, 18.1049137, 17.9043865, 17.7007771, 17.4942074, 17.2847958, 17.0726624, 16.8579330, 16.6407299, 16.4211769, 16.1994019, 15.9755316, 15.7496929, 15.5220127, 15.2926216, 15.0616484, 14.8292227, 14.5954733, 14.3605318, 14.1245279, 13.8875904, 13.6498508, 13.4114380, 13.1724815, 12.9331112, 12.6934528, 12.4536362, 12.2137880, 11.9740334, 11.7344980, 11.4953060, 11.2565804, 11.0184422, 10.7810116, 10.5444088, 10.3087492, 10.0741491, 9.84072304, 9.60858154, 9.37783623, 9.14859390, 8.92096138, 8.69504261, 8.47093868, 8.24874783, 8.02856827, 7.81049252, 7.59461355, 7.38101959, 7.16979694, 6.96102858, 6.75479555, 6.55117512, 6.35024118, 6.15206575, 5.95671749, 5.76426077, 5.57475805, 5.38826799, 5.20484591, 5.02454376, 4.84741068, 4.67349148, 4.50282812, 4.33545971, 4.17142105, 4.01074362, 3.85345531, 3.69958138, 3.54914308, 3.40215802, 3.25864053, 3.11860204, 2.98204994, 2.84898877, 2.71941948, 2.59333992, 2.47074485, 2.35162568, 2.23597097, 2.12376618, 2.01499391, 1.90963376, 1.80766296, 1.70905590, 1.61378443, 1.52181816, 1.43312442, 1.34766829, 1.26541305, 1.18631995, 1.11034870, 1.03745759, 0.967603266, 0.900741696, 0.836827576, 0.775815010, 0.717657566, 0.662308574, 0.609721243, 0.559849143, 0.512646079, 0.468066841, 0.426067024, 0.386603683, 0.349635392, 0.315122664, 0.283028334, 0.253317773, 0.225959271, 0.200924486, 0.178188711, 0.157731339, 0.139536217, 0.123592108, 0.109893061, 9.84388962E-02, 8.92356560E-02, 8.22960511E-02, 7.76399821E-02, 7.52949938E-02])
    curidt = np.array([1.13686838E-13, 2343.58984, 7030.22021, 14058.7920, 23427.6582, 35134.6211, 49176.9336, 65551.2969, 84253.8750, 105280.266, 128625.523, 154284.172, 182250.141, 212516.859, 245077.172, 279923.406, 317047.312, 356440.094, 398092.406, 441994.406, 488135.594, 536505.062, 587091.188, 639881.938, 694864.625, 752026.188, 811352.750, 872830.125, 936443.438, 1002177.31, 1070015.88, 1139942.50, 1211940.25, 1285991.62, 1362078.38, 1440182.00, 1520283.00, 1602361.75, 1686398.00, 1772370.75, 1860258.63, 1950039.63, 2041691.38, 2135190.50, 2230513.75, 2327636.50, 2426534.50, 2527182.25, 2629554.00, 2733623.25, 2839363.25, 2946746.50, 3055744.75, 3166329.75, 3278472.50, 3392142.75, 3507310.75, 3623945.75, 3742016.25, 3861490.50, 3982336.00, 4104519.75, 4228008.50, 4352768.00, 4478763.50, 4605960.00, 4734322.00, 4863813.00, 4994396.00, 5126034.50, 5258689.50, 5392323.00, 5526896.00, 5662369.50, 5798702.50, 5935854.50, 6073785.00, 6212451.50, 6351812.50, 6491824.00, 6632443.50, 6773627.00, 6915329.50, 7057506.50, 7200112.00, 7343100.00, 7486424.00, 7630036.50, 7773890.00, 7917935.50, 8062125.00, 8206408.50, 8350736.00, 8495057.00, 8639321.00, 8783476.00, 8927469.00, 9071249.00, 9214761.00, 9357952.00, 9500767.00, 9643152.00, 9785051.00, 9926408.00, 10067167.0, 10207269.0, 10346659.0, 10485277.0, 10623064.0, 10759962.0, 10895910.0, 11030849.0, 11164716.0, 11297451.0, 11428991.0, 11559274.0, 11688237.0, 11815815.0, 11941946.0, 12066563.0, 12189602.0, 12310996.0, 12430679.0, 12548584.0, 12664644.0, 12778790.0, 12890954.0, 13001067.0, 13109058.0, 13214858.0, 13318395.0, 13419598.0, 13518396.0, 13614715.0, 13708483.0, 13799626.0, 13888070.0, 13973740.0, 14056561.0, 14136457.0, 14213351.0, 14287168.0, 14357829.0, 14425256.0, 14489371.0, 14550095.0, 14607348.0, 14661050.0, 14711119.0, 14757476.0, 14800038.0, 14838722.0, 14873446.0, 14904126.0, 14930678.0, 14953018.0, 14971060.0, 14984719.0, 14993908.0, 14998541.0])

    rmaj = 6.2
    triang = 0.0
    ellip = 1.0
    rhodt = np.linspace(0.00625, 1.99375, len(presdt))
    p0in = np.array([0.1, 0.01, 1.0])
    ps, solv = uvmoms(rmaj, triang, ellip, rhodt, presdt, curidt, p0in, ivp_method = 'RK45', rt_method = 'lm')

    print(solv[:, [0, -1]])

    import matplotlib.pyplot as plt
    n2 = len(solv[0])
    rr = rhodt[-1] * np.array([[solv[0][k] - ps[k] * np.cos(t) + solv[2][k] * np.cos(2*t) for t in np.linspace(0, 2*np.pi, 64)] for k in range(n2)])
    zz = rhodt[-1] * np.array([[solv[4][k] * (ps[k] * np.sin(t) + solv[2][k] * np.sin(2*t)) for t in np.linspace(0, 2*np.pi, 64)] for k in range(n2)])
    plt.axes().set_aspect('equal')
    for j in range(0, len(rr), len(rr) // 12):
        plt.plot(rr[j,:], zz[j,:])
    plt.show()
