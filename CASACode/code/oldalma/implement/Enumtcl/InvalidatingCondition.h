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


#if     !defined(_INVALIDATINGCONDITION_H)

#include <CInvalidatingCondition.h>
#define _INVALIDATINGCONDITION_H
#endif 

#if     !defined(_INVALIDATINGCONDITION_HH)

#include "Enum.hpp"

using namespace InvalidatingConditionMod;

template<>
 struct enum_set_traits<InvalidatingCondition> : public enum_set_traiter<InvalidatingCondition,5,InvalidatingConditionMod::RECEIVER_POWER_DOWN> {};

template<>
class enum_map_traits<InvalidatingCondition,void> : public enum_map_traiter<InvalidatingCondition,void> {
public:
  static bool   init_;
  static string typeName_;
  static string enumerationDesc_;
  static string order_;
  static string xsdBaseType_;
  static bool   init(){
    EnumPar<void> ep;
    m_.insert(pair<InvalidatingCondition,EnumPar<void> >
     (InvalidatingConditionMod::ANTENNA_DISCONNECT,ep((int)InvalidatingConditionMod::ANTENNA_DISCONNECT,"ANTENNA_DISCONNECT","Antenna was disconnected")));
    m_.insert(pair<InvalidatingCondition,EnumPar<void> >
     (InvalidatingConditionMod::ANTENNA_MOVE,ep((int)InvalidatingConditionMod::ANTENNA_MOVE,"ANTENNA_MOVE","Antenna was moved")));
    m_.insert(pair<InvalidatingCondition,EnumPar<void> >
     (InvalidatingConditionMod::ANTENNA_POWER_DOWN,ep((int)InvalidatingConditionMod::ANTENNA_POWER_DOWN,"ANTENNA_POWER_DOWN","Antenna was powered down")));
    m_.insert(pair<InvalidatingCondition,EnumPar<void> >
     (InvalidatingConditionMod::RECEIVER_EXCHANGE,ep((int)InvalidatingConditionMod::RECEIVER_EXCHANGE,"RECEIVER_EXCHANGE","Receiver was exchanged")));
    m_.insert(pair<InvalidatingCondition,EnumPar<void> >
     (InvalidatingConditionMod::RECEIVER_POWER_DOWN,ep((int)InvalidatingConditionMod::RECEIVER_POWER_DOWN,"RECEIVER_POWER_DOWN","Receiver was powered down")));
    return true;
  }
  static map<InvalidatingCondition,EnumPar<void> > m_;
};
#define _INVALIDATINGCONDITION_HH
#endif