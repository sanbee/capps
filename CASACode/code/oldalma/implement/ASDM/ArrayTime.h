/*
 * ALMA - Atacama Large Millimeter Array
 * (c) European Southern Observatory, 2002
 * (c) Associated Universities Inc., 2002
 * Copyright by ESO (in the framework of the ALMA collaboration),
 * Copyright by AUI (in the framework of the ALMA collaboration),
 * All rights reserved.
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY, without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111-1307  USA
 *
 * File ArrayTime.h
 */

#ifndef ArrayTime_CLASS
#define ArrayTime_CLASS

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <Interval.h> 
#include <UTCCorrection.h>

#ifndef WITHOUT_ACS
#include <asdmIDLTypesC.h>
using asdmIDLTypes::IDLArrayTime;
#endif

#include "EndianStream.h"
using asdm::EndianOSStream;
using asdm::EndianISStream;

namespace asdm {

/**
 * The ArrayTime class implements the concept of a point in time, implemented
 * as an Interval of time since 17 November 1858 00:00:00 UTC, the beginning of the 
 * modified Julian Day.
 * <p>
 * All dates are assumed to be in the Gregorian calendar, including those 
 * prior to October 15, 1582.  So, if you are interested in very old dates, 
 * this isn't the most convenient class to use. 
 * <p>
 * Internally the time is kept in units of nanoseconds (10<sup>-9</sup> seconds).
 * The base time is 17 November 1858 00:00:00 UTC, and the maximum time is to the
 * year 2151 (2151-02-25T23:47:16.854775807).  This differs from the OMG Time service 
 * The OMG time is in units of 100 nanoseconds using the beginning of the Gregorian
 * calandar,15 October 1582 00:00:00 UTC, as the base time.
 * The reason for this increased accuracy is that the Control system is capable of 
 * measuring time to an accuracy of 40 nanoseconds.  Therefore, by adhering to the 
 * representation of time used in the OMG Time Serivce we would be losing precision.
 * <p>
 * The Time class is an extension of the Interval class, since all times
 * are intervals since 17 November 1858 00:00:00 UTC.
 * <p>
 * All times in this class are assumed to be International 
 * Atomic Time (TAI).  A specific TAI time differs from the corresponding 
 * UTC time by an offset that is an integral number of seconds.  
 * <p>
 * In the methods that give various quantities associated with
 * calendar times, this class does not apply any UTC corrections. 
 * Therefore, if you use these methods to produce calendar times, the 
 * results will differ from civil time by a few seconds.  The classes 
 * UTCTime and LocalTime take the UTC and timezone corrections into 
 * account.
 * <p>
 * The main reference used in crafting these methods is 
 * Astronomical Algorithms by Jean Meeus, second edition,
 * 2000, Willmann-Bell, Inc., ISBN 0-943396-61-1.  See
 * chapter 7, "Julian day", and chapter 12, "Sidereal Time".
 * 
 * @version 1.0 12 January, 2004
 * @author Allen Farris
 * @version 1.1 Aug 8, 2006
 * @author Michel Caillat 
 * added toBin/fromBin methods.
 */
class ArrayTime : public Interval {

public:

	// Useful constants
	const static int numberSigDigitsInASecond 					= 9;
	const static long long unitsInASecond 						= 1000000000LL;
	const static long long unitsInADayL 						= 86400000000000LL;
	const static double unitsInADay ;	
	const static double unitsInADayDiv100 ;		
	const static double julianDayOfBase ;	
	const static long long julianDayOfBaseInUnitsInADayDiv100 	= 2073600432000000000LL;

	static  bool isLeapYear(int year);
	static  double getMJD(double jd);
	static  double getJD(double mjd);
	static  ArrayTime add(const ArrayTime &time, const Interval &interval);
	static  ArrayTime sub(const ArrayTime &time, const Interval &interval) ;
	static  ArrayTime getArrayTime(StringTokenizer &t) ;

	ArrayTime(); 
	ArrayTime (const string &s); 
	ArrayTime(const ArrayTime &t);
#ifndef WITHOUT_ACS 
	ArrayTime (const IDLArrayTime &t);
#endif 
	ArrayTime(int year, int month, double day); 
	ArrayTime(int year, int month, int day, int hour, int minute, double second);
	ArrayTime(double modifiedJulianDay);
	ArrayTime(int modifiedJulianDay, double secondsInADay);
	ArrayTime(long long nanoseconds); 

	double getJD() const; 
	double getMJD() const; 
	double getJDI() const; 
	double getMJDI() const; 

#ifndef WITHOUT_ACS
	IDLArrayTime toIDLArrayTime() const; 
#endif
	string toFITS() const; 
	
			
	/**
	 * Write the binary representation of this to a EndianOSStream.
	 */		
	void toBin(EndianOSStream& eoss);

	/**
	 * Write the binary representation of a vector of ArrayTime to a EndianOSStream.
	 * @param arrayTime the vector of ArrayTime to be written
	 * @param eoss the EndianOSStream to be written to
	 */
	static void toBin(const vector<ArrayTime>& arrayTime,  EndianOSStream& eoss);
	
	/**
	 * Write the binary representation of a vector of vector of ArrayTime to a EndianOSStream.
	 * @param arrayTime the vector of vector of ArrayTime to be written
	 * @param eoss the EndianOSStream to be written to
	 */	
	static void toBin(const vector<vector<ArrayTime> >& arrayTime,  EndianOSStream& eoss);
	
	/**
	 * Write the binary representation of a vector of vector of vector of ArrayTime to a EndianOSStream.
	 * @param arrayTime the vector of vector of vector of ArrayTime to be written
	 * @param eoss the EndianOSStream to be written to
	 */
	static void toBin(const vector<vector<vector<ArrayTime> > >& arrayTime,  EndianOSStream& eoss);

	/**
	 * Read the binary representation of an ArrayTime from a EndianISStream
	 * and use the read value to set an  ArrayTime.
	 * @param eiss the EndianStream to be read
	 * @return an ArrayTime
	 */
	static ArrayTime fromBin(EndianISStream& eiss);
	
	/**
	 * Read the binary representation of  a vector of  ArrayTime from an EndianISStream
	 * and use the read value to set a vector of  ArrayTime.
	 * @param dis the EndianISStream to be read
	 * @return a vector of ArrayTime
	 */	 
	 static vector<ArrayTime> from1DBin(EndianISStream & eiss);
	 
	/**
	 * Read the binary representation of  a vector of vector of ArrayTime from an EndianISStream
	 * and use the read value to set a vector of  vector of ArrayTime.
	 * @param eiis the EndianISStream to be read
	 * @return a vector of vector of ArrayTime
	 */	 
	 static vector<vector<ArrayTime> > from2DBin(EndianISStream & eiss);
	 
	/**
	 * Read the binary representation of  a vector of vector of vector of ArrayTime from an EndianISStream
	 * and use the read value to set a vector of  vector of vector of ArrayTime.
	 * @param eiss the EndianISStream to be read
	 * @return a vector of vector of vector of ArrayTime
	 */	 
	 static vector<vector<vector<ArrayTime> > > from3DBin(EndianISStream & eiss);	
	 	
	int *getDateTime() const; 
	double getTimeOfDay() const; 
	int getDayOfWeek() const; 
	int getDayOfYear() const; 
	string timeOfDayToString() const;
	double getLocalSiderealTime(double longitudeInHours) const; 
	double getGreenwichMeanSiderealTime() const; 

	static double unitToJD(long long unit); 
	static double unitToMJD(long long unit);
	static long long jdToUnit(double jd); 
	static long long mjdToUnit(double mjd); 

	static double utcCorrection(double jd); 

private:

	static long long init(int year, int month, double day); 
	static long long init(int year, int month, int day, int hour, int minute, double second);
	long long FITSString(string t) const; 

/*
	static const int numberSigDigitsInASecond;
	static const long long unitsInASecond;
	static const long long unitsInADayL;
	static const double unitsInADay;
	static const double unitsInADayDiv100;
	static const double julianDayOfBase;
	static const long long julianDayOfBaseInUnitsInADayDiv100;
*/
	static const UTCCorrection *UTCCorrectionTable;
	static const UTCCorrection UTCLast;

};

inline double ArrayTime::getJDI() const {return getJD();}
inline double ArrayTime::getMJDI()const  {return getMJD();}
} // End namespace asdm

#endif /* ArrayTime_CLASS */