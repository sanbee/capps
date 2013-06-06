
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
 * Warning!
 *  -------------------------------------------------------------------- 
 * | This is generated code!  Do not modify this file.                  |
 * | If you do, all changes will be lost when the file is re-generated. |
 *  --------------------------------------------------------------------
 *
 * File GainTrackingRow.h
 */
 
#ifndef GainTrackingRow_CLASS
#define GainTrackingRow_CLASS

#include <vector>
#include <string>
#include <set>
using std::vector;
using std::string;
using std::set;

#ifndef WITHOUT_ACS
#include <asdmIDLC.h>
using asdmIDL::GainTrackingRowIDL;
#endif

#include <Angle.h>
#include <AngularRate.h>
#include <ArrayTime.h>
#include <ArrayTimeInterval.h>
#include <Complex.h>
#include <Entity.h>
#include <EntityId.h>
#include <EntityRef.h>
#include <Flux.h>
#include <Frequency.h>
#include <Humidity.h>
#include <Interval.h>
#include <Length.h>
#include <Pressure.h>
#include <Speed.h>
#include <Tag.h>
#include <Temperature.h>
#include <ConversionException.h>
#include <NoSuchRow.h>
#include <IllegalAccessException.h>

/*
#include <Enumerations.h>
using namespace enumerations;
 */




	

	

	

	

	

	

	

	

	

	



using asdm::Angle;
using asdm::AngularRate;
using asdm::ArrayTime;
using asdm::Complex;
using asdm::Entity;
using asdm::EntityId;
using asdm::EntityRef;
using asdm::Flux;
using asdm::Frequency;
using asdm::Humidity;
using asdm::Interval;
using asdm::Length;
using asdm::Pressure;
using asdm::Speed;
using asdm::Tag;
using asdm::Temperature;
using asdm::ConversionException;
using asdm::NoSuchRow;
using asdm::IllegalAccessException;

/*\file GainTracking.h
    \brief Generated from model's revision "1.46", branch "HEAD"
*/

namespace asdm {

//class asdm::GainTrackingTable;


// class asdm::SpectralWindowRow;
class SpectralWindowRow;

// class asdm::AntennaRow;
class AntennaRow;

// class asdm::FeedRow;
class FeedRow;
	

/**
 * The GainTrackingRow class is a row of a GainTrackingTable.
 * 
 * Generated from model's revision "1.46", branch "HEAD"
 *
 */
class GainTrackingRow {
friend class asdm::GainTrackingTable;

public:

	virtual ~GainTrackingRow();

	/**
	 * Return the table to which this row belongs.
	 */
	GainTrackingTable &getTable() const;
	
#ifndef WITHOUT_ACS
	/**
	 * Return this row in the form of an IDL struct.
	 * @return The values of this row as a GainTrackingRowIDL struct.
	 */
	GainTrackingRowIDL *toIDL() const;
#endif
	
#ifndef WITHOUT_ACS
	/**
	 * Fill the values of this row from the IDL struct GainTrackingRowIDL.
	 * @param x The IDL struct containing the values used to fill this row.
	 */
	void setFromIDL (GainTrackingRowIDL x) throw(ConversionException);
#endif
	
	/**
	 * Return this row in the form of an XML string.
	 * @return The values of this row as an XML string.
	 */
	string toXML() const;

	/**
	 * Fill the values of this row from an XML string 
	 * that was produced by the toXML() method.
	 * @param x The XML string being used to set the values of this row.
	 */
	void setFromXML (string rowDoc) throw(ConversionException);
	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute timeInterval
	
	
	

	
 	/**
 	 * Get timeInterval.
 	 * @return timeInterval as ArrayTimeInterval
 	 */
 	ArrayTimeInterval getTimeInterval() const;
	
 
 	
 	
 	/**
 	 * Set timeInterval with the specified ArrayTimeInterval.
 	 * @param timeInterval The ArrayTimeInterval value to which timeInterval is to be set.
 	 
 		
 			
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 		
 	 */
 	void setTimeInterval (ArrayTimeInterval timeInterval);
  		
	
	
	


	
	// ===> Attribute attenuator
	
	
	

	
 	/**
 	 * Get attenuator.
 	 * @return attenuator as float
 	 */
 	float getAttenuator() const;
	
 
 	
 	
 	/**
 	 * Set attenuator with the specified float.
 	 * @param attenuator The float value to which attenuator is to be set.
 	 
 		
 			
 	 */
 	void setAttenuator (float attenuator);
  		
	
	
	


	
	// ===> Attribute samplingLevel, which is optional
	
	
	
	/**
	 * The attribute samplingLevel is optional. Return true if this attribute exists.
	 * @return true if and only if the samplingLevel attribute exists. 
	 */
	bool isSamplingLevelExists() const;
	

	
 	/**
 	 * Get samplingLevel, which is optional.
 	 * @return samplingLevel as float
 	 * @throws IllegalAccessException If samplingLevel does not exist.
 	 */
 	float getSamplingLevel() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set samplingLevel with the specified float.
 	 * @param samplingLevel The float value to which samplingLevel is to be set.
 	 
 		
 	 */
 	void setSamplingLevel (float samplingLevel);
		
	
	
	
	/**
	 * Mark samplingLevel, which is an optional field, as non-existent.
	 */
	void clearSamplingLevel ();
	


	
	// ===> Attribute delayoff1
	
	
	

	
 	/**
 	 * Get delayoff1.
 	 * @return delayoff1 as Interval
 	 */
 	Interval getDelayoff1() const;
	
 
 	
 	
 	/**
 	 * Set delayoff1 with the specified Interval.
 	 * @param delayoff1 The Interval value to which delayoff1 is to be set.
 	 
 		
 			
 	 */
 	void setDelayoff1 (Interval delayoff1);
  		
	
	
	


	
	// ===> Attribute delayoff2
	
	
	

	
 	/**
 	 * Get delayoff2.
 	 * @return delayoff2 as Interval
 	 */
 	Interval getDelayoff2() const;
	
 
 	
 	
 	/**
 	 * Set delayoff2 with the specified Interval.
 	 * @param delayoff2 The Interval value to which delayoff2 is to be set.
 	 
 		
 			
 	 */
 	void setDelayoff2 (Interval delayoff2);
  		
	
	
	


	
	// ===> Attribute phaseoff1
	
	
	

	
 	/**
 	 * Get phaseoff1.
 	 * @return phaseoff1 as Angle
 	 */
 	Angle getPhaseoff1() const;
	
 
 	
 	
 	/**
 	 * Set phaseoff1 with the specified Angle.
 	 * @param phaseoff1 The Angle value to which phaseoff1 is to be set.
 	 
 		
 			
 	 */
 	void setPhaseoff1 (Angle phaseoff1);
  		
	
	
	


	
	// ===> Attribute phaseoff2
	
	
	

	
 	/**
 	 * Get phaseoff2.
 	 * @return phaseoff2 as Angle
 	 */
 	Angle getPhaseoff2() const;
	
 
 	
 	
 	/**
 	 * Set phaseoff2 with the specified Angle.
 	 * @param phaseoff2 The Angle value to which phaseoff2 is to be set.
 	 
 		
 			
 	 */
 	void setPhaseoff2 (Angle phaseoff2);
  		
	
	
	


	
	// ===> Attribute rateoff1
	
	
	

	
 	/**
 	 * Get rateoff1.
 	 * @return rateoff1 as AngularRate
 	 */
 	AngularRate getRateoff1() const;
	
 
 	
 	
 	/**
 	 * Set rateoff1 with the specified AngularRate.
 	 * @param rateoff1 The AngularRate value to which rateoff1 is to be set.
 	 
 		
 			
 	 */
 	void setRateoff1 (AngularRate rateoff1);
  		
	
	
	


	
	// ===> Attribute rateoff2
	
	
	

	
 	/**
 	 * Get rateoff2.
 	 * @return rateoff2 as AngularRate
 	 */
 	AngularRate getRateoff2() const;
	
 
 	
 	
 	/**
 	 * Set rateoff2 with the specified AngularRate.
 	 * @param rateoff2 The AngularRate value to which rateoff2 is to be set.
 	 
 		
 			
 	 */
 	void setRateoff2 (AngularRate rateoff2);
  		
	
	
	


	
	// ===> Attribute phaseRefOffset, which is optional
	
	
	
	/**
	 * The attribute phaseRefOffset is optional. Return true if this attribute exists.
	 * @return true if and only if the phaseRefOffset attribute exists. 
	 */
	bool isPhaseRefOffsetExists() const;
	

	
 	/**
 	 * Get phaseRefOffset, which is optional.
 	 * @return phaseRefOffset as Angle
 	 * @throws IllegalAccessException If phaseRefOffset does not exist.
 	 */
 	Angle getPhaseRefOffset() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set phaseRefOffset with the specified Angle.
 	 * @param phaseRefOffset The Angle value to which phaseRefOffset is to be set.
 	 
 		
 	 */
 	void setPhaseRefOffset (Angle phaseRefOffset);
		
	
	
	
	/**
	 * Mark phaseRefOffset, which is an optional field, as non-existent.
	 */
	void clearPhaseRefOffset ();
	


	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute antennaId
	
	
	

	
 	/**
 	 * Get antennaId.
 	 * @return antennaId as Tag
 	 */
 	Tag getAntennaId() const;
	
 
 	
 	
 	/**
 	 * Set antennaId with the specified Tag.
 	 * @param antennaId The Tag value to which antennaId is to be set.
 	 
 		
 			
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 		
 	 */
 	void setAntennaId (Tag antennaId);
  		
	
	
	


	
	// ===> Attribute feedId
	
	
	

	
 	/**
 	 * Get feedId.
 	 * @return feedId as int
 	 */
 	int getFeedId() const;
	
 
 	
 	
 	/**
 	 * Set feedId with the specified int.
 	 * @param feedId The int value to which feedId is to be set.
 	 
 		
 			
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 		
 	 */
 	void setFeedId (int feedId);
  		
	
	
	


	
	// ===> Attribute spectralWindowId
	
	
	

	
 	/**
 	 * Get spectralWindowId.
 	 * @return spectralWindowId as Tag
 	 */
 	Tag getSpectralWindowId() const;
	
 
 	
 	
 	/**
 	 * Set spectralWindowId with the specified Tag.
 	 * @param spectralWindowId The Tag value to which spectralWindowId is to be set.
 	 
 		
 			
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 		
 	 */
 	void setSpectralWindowId (Tag spectralWindowId);
  		
	
	
	


	///////////
	// Links //
	///////////
	
	

	
		
	/**
	 * spectralWindowId pointer to the row in the SpectralWindow table having SpectralWindow.spectralWindowId == spectralWindowId
	 * @return a SpectralWindowRow*
	 * 
	 
	 */
	 SpectralWindowRow* getSpectralWindowUsingSpectralWindowId();
	 

	

	

	
		
	/**
	 * antennaId pointer to the row in the Antenna table having Antenna.antennaId == antennaId
	 * @return a AntennaRow*
	 * 
	 
	 */
	 AntennaRow* getAntennaUsingAntennaId();
	 

	

	

	
		
	// ===> Slice link from a row of GainTracking table to a collection of row of Feed table.
	
	/**
	 * Get the collection of row in the Feed table having feedId == this.feedId
	 * 
	 * @return a vector of FeedRow *
	 */
	vector <FeedRow *> getFeeds();
	
	

	

	
	
	
	/**
	 * Compare each mandatory attribute except the autoincrementable one of this GainTrackingRow with 
	 * the corresponding parameters and return true if there is a match and false otherwise.
	 */ 
	bool compareNoAutoInc(Tag antennaId, int feedId, Tag spectralWindowId, ArrayTimeInterval timeInterval, float attenuator, Interval delayoff1, Interval delayoff2, Angle phaseoff1, Angle phaseoff2, AngularRate rateoff1, AngularRate rateoff2);
	
	

	
	bool compareRequiredValue(float attenuator, Interval delayoff1, Interval delayoff2, Angle phaseoff1, Angle phaseoff2, AngularRate rateoff1, AngularRate rateoff2); 
		 
	
	/**
	 * Return true if all required attributes of the value part are equal to their homologues
	 * in x and false otherwise.
	 *
	 * @param x a pointer on the GainTrackingRow whose required attributes of the value part 
	 * will be compared with those of this.
	 * @return a boolean.
	 */
	bool equalByRequiredValue(GainTrackingRow* x) ;

private:
	/**
	 * The table to which this row belongs.
	 */
	GainTrackingTable &table;
	/**
	 * Whether this row has been added to the table or not.
	 */
	bool hasBeenAdded;

	// This method is used by the Table class when this row is added to the table.
	void isAdded();


	/**
	 * Create a GainTrackingRow.
	 * <p>
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 */ 
	GainTrackingRow (GainTrackingTable &table);

	/**
	 * Create a GainTrackingRow using a copy constructor mechanism.
	 * <p>
	 * Given a GainTrackingRow row and a GainTrackingTable table, the method creates a new
	 * GainTrackingRow owned by table. Each attribute of the created row is a copy (deep)
	 * of the corresponding attribute of row. The method does not add the created
	 * row to its table, its simply parents it to table, a call to the add method
	 * has to be done in order to get the row added (very likely after having modified
	 * some of its attributes).
	 * If row is null then the method returns a row with default values for its attributes. 
	 *
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 * @param row  The row which is to be copied.
	 */
	 GainTrackingRow (GainTrackingTable &table, GainTrackingRow &row);
	 	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute timeInterval
	
	

	ArrayTimeInterval timeInterval;

	
	
 	

	
	// ===> Attribute attenuator
	
	

	float attenuator;

	
	
 	

	
	// ===> Attribute samplingLevel, which is optional
	
	
	bool samplingLevelExists;
	

	float samplingLevel;

	
	
 	

	
	// ===> Attribute delayoff1
	
	

	Interval delayoff1;

	
	
 	

	
	// ===> Attribute delayoff2
	
	

	Interval delayoff2;

	
	
 	

	
	// ===> Attribute phaseoff1
	
	

	Angle phaseoff1;

	
	
 	

	
	// ===> Attribute phaseoff2
	
	

	Angle phaseoff2;

	
	
 	

	
	// ===> Attribute rateoff1
	
	

	AngularRate rateoff1;

	
	
 	

	
	// ===> Attribute rateoff2
	
	

	AngularRate rateoff2;

	
	
 	

	
	// ===> Attribute phaseRefOffset, which is optional
	
	
	bool phaseRefOffsetExists;
	

	Angle phaseRefOffset;

	
	
 	

	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute antennaId
	
	

	Tag antennaId;

	
	
 	

	
	// ===> Attribute feedId
	
	

	int feedId;

	
	
 	

	
	// ===> Attribute spectralWindowId
	
	

	Tag spectralWindowId;

	
	
 	

	///////////
	// Links //
	///////////
	
	
		

	 

	

	
		

	 

	

	
		


	


};

} // End namespace asdm

#endif /* GainTracking_CLASS */