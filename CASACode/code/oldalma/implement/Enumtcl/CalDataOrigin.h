/*
 *
 * /////////////////////////////////////////////////////////////////
 * // WARNING!  DO NOT MODIFY THIS FILE!                          //
 * //  ---------------------------------------------------------  //
 * // | This is generated code using a C++ template function!   | //
 * // ! Do not modify this file.                                | //
 * // | Any changes will be lost when the file is re-generated. | //
 * //  ---------------------------------------------------------  //
 * /////////////////////////////////////////////////////////////////
 *
 */


#if     !defined(_CALDATAORIGIN_H)

#include <CCalDataOrigin.h>
#define _CALDATAORIGIN_H
#endif 

#if     !defined(_CALDATAORIGIN_HH)

#include "Enum.hpp"

using namespace CalDataOriginMod;

template<>
 struct enum_set_traits<CalDataOrigin> : public enum_set_traiter<CalDataOrigin,8,CalDataOriginMod::HOLOGRAPHY> {};

template<>
class enum_map_traits<CalDataOrigin,void> : public enum_map_traiter<CalDataOrigin,void> {
public:
  static bool   init_;
  static string typeName_;
  static string enumerationDesc_;
  static string order_;
  static string xsdBaseType_;
  static bool   init(){
    EnumPar<void> ep;
    m_.insert(pair<CalDataOrigin,EnumPar<void> >
     (CalDataOriginMod::TOTAL_POWER,ep((int)CalDataOriginMod::TOTAL_POWER,"TOTAL_POWER","Total Power data (from detectors)")));
    m_.insert(pair<CalDataOrigin,EnumPar<void> >
     (CalDataOriginMod::WVR,ep((int)CalDataOriginMod::WVR,"WVR","Water vapour radiometrers")));
    m_.insert(pair<CalDataOrigin,EnumPar<void> >
     (CalDataOriginMod::CHANNEL_AVERAGE_AUTO,ep((int)CalDataOriginMod::CHANNEL_AVERAGE_AUTO,"CHANNEL_AVERAGE_AUTO","Autocorrelations from channel average data")));
    m_.insert(pair<CalDataOrigin,EnumPar<void> >
     (CalDataOriginMod::CHANNEL_AVERAGE_CROSS,ep((int)CalDataOriginMod::CHANNEL_AVERAGE_CROSS,"CHANNEL_AVERAGE_CROSS","Crosscorrelations from channel average data")));
    m_.insert(pair<CalDataOrigin,EnumPar<void> >
     (CalDataOriginMod::FULL_RESOLUTION_AUTO,ep((int)CalDataOriginMod::FULL_RESOLUTION_AUTO,"FULL_RESOLUTION_AUTO","Autocorrelations from full-resolution data")));
    m_.insert(pair<CalDataOrigin,EnumPar<void> >
     (CalDataOriginMod::FULL_RESOLUTION_CROSS,ep((int)CalDataOriginMod::FULL_RESOLUTION_CROSS,"FULL_RESOLUTION_CROSS","Cross correlations from full-resolution data")));
    m_.insert(pair<CalDataOrigin,EnumPar<void> >
     (CalDataOriginMod::OPTICAL_POINTING,ep((int)CalDataOriginMod::OPTICAL_POINTING,"OPTICAL_POINTING","Optical pointing data")));
    m_.insert(pair<CalDataOrigin,EnumPar<void> >
     (CalDataOriginMod::HOLOGRAPHY,ep((int)CalDataOriginMod::HOLOGRAPHY,"HOLOGRAPHY","data from holography receivers")));
    return true;
  }
  static map<CalDataOrigin,EnumPar<void> > m_;
};
#define _CALDATAORIGIN_HH
#endif