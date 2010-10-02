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


#if     !defined(_ANTENNAMOTIONPATTERN_H)

#include <CAntennaMotionPattern.h>
#define _ANTENNAMOTIONPATTERN_H
#endif 

#if     !defined(_ANTENNAMOTIONPATTERN_HH)

#include "Enum.hpp"

using namespace AntennaMotionPatternMod;

template<>
 struct enum_set_traits<AntennaMotionPattern> : public enum_set_traiter<AntennaMotionPattern,9,AntennaMotionPatternMod::UNSPECIFIED> {};

template<>
class enum_map_traits<AntennaMotionPattern,void> : public enum_map_traiter<AntennaMotionPattern,void> {
public:
  static bool   init_;
  static string typeName_;
  static string enumerationDesc_;
  static string order_;
  static string xsdBaseType_;
  static bool   init(){
    EnumPar<void> ep;
    m_.insert(pair<AntennaMotionPattern,EnumPar<void> >
     (AntennaMotionPatternMod::NONE,ep((int)AntennaMotionPatternMod::NONE,"NONE","un-documented")));
    m_.insert(pair<AntennaMotionPattern,EnumPar<void> >
     (AntennaMotionPatternMod::CROSS_SCAN,ep((int)AntennaMotionPatternMod::CROSS_SCAN,"CROSS_SCAN","un-documented")));
    m_.insert(pair<AntennaMotionPattern,EnumPar<void> >
     (AntennaMotionPatternMod::SPIRAL,ep((int)AntennaMotionPatternMod::SPIRAL,"SPIRAL","un-documented")));
    m_.insert(pair<AntennaMotionPattern,EnumPar<void> >
     (AntennaMotionPatternMod::CIRCLE,ep((int)AntennaMotionPatternMod::CIRCLE,"CIRCLE","un-documented")));
    m_.insert(pair<AntennaMotionPattern,EnumPar<void> >
     (AntennaMotionPatternMod::THREE_POINTS,ep((int)AntennaMotionPatternMod::THREE_POINTS,"THREE_POINTS","un-documented")));
    m_.insert(pair<AntennaMotionPattern,EnumPar<void> >
     (AntennaMotionPatternMod::FOUR_POINTS,ep((int)AntennaMotionPatternMod::FOUR_POINTS,"FOUR_POINTS","un-documented")));
    m_.insert(pair<AntennaMotionPattern,EnumPar<void> >
     (AntennaMotionPatternMod::FIVE_POINTS,ep((int)AntennaMotionPatternMod::FIVE_POINTS,"FIVE_POINTS","un-documented")));
    m_.insert(pair<AntennaMotionPattern,EnumPar<void> >
     (AntennaMotionPatternMod::TEST,ep((int)AntennaMotionPatternMod::TEST,"TEST","un-documented")));
    m_.insert(pair<AntennaMotionPattern,EnumPar<void> >
     (AntennaMotionPatternMod::UNSPECIFIED,ep((int)AntennaMotionPatternMod::UNSPECIFIED,"UNSPECIFIED","un-documented")));
    return true;
  }
  static map<AntennaMotionPattern,EnumPar<void> > m_;
};
#define _ANTENNAMOTIONPATTERN_HH
#endif
