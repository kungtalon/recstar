package com.meituan.mtpt.rec.tools;

import java.util.Arrays;

public class GeohashUtils {
    private static final char[] BASE_32 = new char[]{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
    private static final int[] BASE_32_IDX;
    public static final int MAX_PRECISION = 24;
    private static final int[] BITS = new int[]{16, 8, 4, 2, 1};
    private static final double[] hashLenToLatHeight;
    private static final double[] hashLenToLonWidth;

    private GeohashUtils() {
    }

    public static String encodeLatLon(double latitude, double longitude) {
        return encodeLatLon(latitude, longitude, 12);
    }

    public static String encodeLatLon(double latitude, double longitude, int precision) {
        double[] latInterval = new double[]{-90.0D, 90.0D};
        double[] lngInterval = new double[]{-180.0D, 180.0D};
        StringBuilder geohash = new StringBuilder(precision);
        boolean isEven = true;
        int bit = 0;
        int ch = 0;

        while(geohash.length() < precision) {
            double mid = 0.0D;
            if (isEven) {
                mid = (lngInterval[0] + lngInterval[1]) / 2.0D;
                if (longitude > mid) {
                    ch |= BITS[bit];
                    lngInterval[0] = mid;
                } else {
                    lngInterval[1] = mid;
                }
            } else {
                mid = (latInterval[0] + latInterval[1]) / 2.0D;
                if (latitude > mid) {
                    ch |= BITS[bit];
                    latInterval[0] = mid;
                } else {
                    latInterval[1] = mid;
                }
            }

            isEven = !isEven;
            if (bit < 4) {
                ++bit;
            } else {
                geohash.append(BASE_32[ch]);
                bit = 0;
                ch = 0;
            }
        }

        return geohash.toString();
    }

    public static String[] getSubGeohashes(String baseGeohash) {
        String[] hashes = new String[BASE_32.length];

        for(int i = 0; i < BASE_32.length; ++i) {
            char c = BASE_32[i];
            hashes[i] = baseGeohash + c;
        }

        return hashes;
    }

    public static double[] lookupDegreesSizeForHashLen(int hashLen) {
        return new double[]{hashLenToLatHeight[hashLen], hashLenToLonWidth[hashLen]};
    }

    public static int lookupHashLenForWidthHeight(double lonErr, double latErr) {
        for(int len = 1; len < 24; ++len) {
            double latHeight = hashLenToLatHeight[len];
            double lonWidth = hashLenToLonWidth[len];
            if (latHeight < latErr && lonWidth < lonErr) {
                return len;
            }
        }

        return 24;
    }

    static {
        BASE_32_IDX = new int[BASE_32[BASE_32.length - 1] - BASE_32[0] + 1];

        assert BASE_32_IDX.length < 100;

        Arrays.fill(BASE_32_IDX, -500);

        for(int i = 0; i < BASE_32.length; BASE_32_IDX[BASE_32[i] - BASE_32[0]] = i++) {
            ;
        }

        hashLenToLatHeight = new double[25];
        hashLenToLonWidth = new double[25];
        hashLenToLatHeight[0] = 180.0D;
        hashLenToLonWidth[0] = 360.0D;
        boolean even = false;

        for(int i = 1; i <= 24; ++i) {
            hashLenToLatHeight[i] = hashLenToLatHeight[i - 1] / (double)(even ? 8 : 4);
            hashLenToLonWidth[i] = hashLenToLonWidth[i - 1] / (double)(even ? 4 : 8);
            even = !even;
        }

    }
}
