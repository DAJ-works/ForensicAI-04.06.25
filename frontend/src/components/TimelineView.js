import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Typography, Box, Paper, Chip, CircularProgress, Button } from '@mui/material';
import { Timeline, TimelineItem, TimelineSeparator, TimelineConnector, 
         TimelineContent, TimelineDot, TimelineOppositeContent } from '@mui/lab';
import PersonIcon from '@mui/icons-material/Person';
import ExitToAppIcon from '@mui/icons-material/ExitToApp';
import MeetingRoomIcon from '@mui/icons-material/MeetingRoom';
import EventIcon from '@mui/icons-material/Event';
import axios from 'axios';

// Event type to icon mapping
const eventIcons = {
  'person_appearance': <PersonIcon />,
  'person_disappearance': <ExitToAppIcon />,
  'person_interaction': <MeetingRoomIcon />,
  'default': <EventIcon />
};

// Event type to color mapping
const eventColors = {
  'person_appearance': 'success',
  'person_disappearance': 'error',
  'person_interaction': 'info',
  'default': 'grey'
};

function TimelineView() {
  const { caseId } = useParams();
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [caseDetails, setCaseDetails] = useState(null);
  const [personDetails, setPersonDetails] = useState({});
  const [filteredEvents, setFilteredEvents] = useState([]);
  const [filters, setFilters] = useState({
    person_appearance: true,
    person_disappearance: true,
    person_interaction: true,
    other: true
  });

  // Fetch timeline data
  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        
        // Get timeline events
        const response = await axios.get(`/api/cases/${caseId}/timeline`);
        setEvents(response.data);
        setFilteredEvents(response.data);
        
        // Get case details
        const caseResponse = await axios.get(`/api/cases/${caseId}`);
        setCaseDetails(caseResponse.data);
        
        // Get person details for all persons involved in events
        const personIds = new Set();
        response.data.forEach(event => {
          if (event.person_id !== undefined) {
            personIds.add(event.person_id);
          }
        });
        
        // Fetch details for each person
        const personData = {};
        for (const personId of personIds) {
          try {
            const personResponse = await axios.get(`/api/cases/${caseId}/persons/${personId}`);
            personData[personId] = personResponse.data;
          } catch (err) {
            console.error(`Error fetching person ${personId}:`, err);
          }
        }
        
        setPersonDetails(personData);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching timeline data:', err);
        setError('Failed to load timeline data');
        setLoading(false);
      }
    }
    
    fetchData();
  }, [caseId]);

  // Apply filters when filters change
  useEffect(() => {
    if (events.length > 0) {
      const filtered = events.filter(event => {
        const eventType = event.event_type || 'other';
        if (eventType === 'person_appearance' && !filters.person_appearance) return false;
        if (eventType === 'person_disappearance' && !filters.person_disappearance) return false;
        if (eventType === 'person_interaction' && !filters.person_interaction) return false;
        if (!['person_appearance', 'person_disappearance', 'person_interaction'].includes(eventType) && !filters.other) return false;
        return true;
      });
      
      setFilteredEvents(filtered);
    }
  }, [filters, events]);

  // Toggle filter status
  const toggleFilter = (filterName) => {
    setFilters({
      ...filters,
      [filterName]: !filters[filterName]
    });
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="70vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="70vh">
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  if (filteredEvents.length === 0) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Timeline - Case {caseId}
        </Typography>
        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <Box sx={{ mb: 2 }}>
            <Typography variant="h6" gutterBottom>Filters</Typography>
            <Chip 
              label="Appearances" 
              color={filters.person_appearance ? "success" : "default"} 
              onClick={() => toggleFilter('person_appearance')}
              sx={{ m: 0.5 }}
            />
            <Chip 
              label="Disappearances" 
              color={filters.person_disappearance ? "error" : "default"} 
              onClick={() => toggleFilter('person_disappearance')}
              sx={{ m: 0.5 }}
            />
            <Chip 
              label="Interactions" 
              color={filters.person_interaction ? "info" : "default"} 
              onClick={() => toggleFilter('person_interaction')}
              sx={{ m: 0.5 }}
            />
            <Chip 
              label="Other Events" 
              color={filters.other ? "warning" : "default"} 
              onClick={() => toggleFilter('other')}
              sx={{ m: 0.5 }}
            />
          </Box>
        </Paper>
        <Typography variant="body1">
          No timeline events match the current filters.
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Timeline - Case {caseId}
      </Typography>
      
      {/* Case details summary */}
      {caseDetails && (
        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            {caseDetails.video_path?.split('/').pop() || 'Video Analysis'}
          </Typography>
          <Typography variant="body2" gutterBottom>
            Analyzed on: {new Date(caseDetails.timestamp).toLocaleString()}
          </Typography>
          <Typography variant="body2" gutterBottom>
            Total detections: {caseDetails.total_detections}
          </Typography>
          <Typography variant="body2" gutterBottom>
            Unique persons: {caseDetails.person_identities?.length || 0}
          </Typography>
          <Typography variant="body2" gutterBottom>
            Timeline events: {filteredEvents.length}
          </Typography>
        </Paper>
      )}
      
      {/* Filters */}
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Box sx={{ mb: 2 }}>
          <Typography variant="h6" gutterBottom>Filters</Typography>
          <Chip 
            label="Appearances" 
            color={filters.person_appearance ? "success" : "default"} 
            onClick={() => toggleFilter('person_appearance')}
            sx={{ m: 0.5 }}
          />
          <Chip 
            label="Disappearances" 
            color={filters.person_disappearance ? "error" : "default"} 
            onClick={() => toggleFilter('person_disappearance')}
            sx={{ m: 0.5 }}
          />
          <Chip 
            label="Interactions" 
            color={filters.person_interaction ? "info" : "default"} 
            onClick={() => toggleFilter('person_interaction')}
            sx={{ m: 0.5 }}
          />
          <Chip 
            label="Other Events" 
            color={filters.other ? "warning" : "default"} 
            onClick={() => toggleFilter('other')}
            sx={{ m: 0.5 }}
          />
        </Box>
      </Paper>
      
      {/* Timeline */}
      <Paper elevation={3} sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>Event Timeline</Typography>
        
        <Timeline position="alternate">
          {filteredEvents.map((event, index) => (
            <TimelineItem key={index}>
              <TimelineOppositeContent color="text.secondary">
                {event.timestamp ? (
                  <>
                    <Typography variant="body2">
                      {new Date(event.timestamp * 1000).toLocaleTimeString()}
                    </Typography>
                    <Typography variant="caption">
                      Frame: {event.frame}
                    </Typography>
                  </>
                ) : (
                  <Typography variant="body2">
                    Unknown time
                  </Typography>
                )}
              </TimelineOppositeContent>
              
              <TimelineSeparator>
                <TimelineDot color={eventColors[event.event_type] || eventColors.default}>
                  {eventIcons[event.event_type] || eventIcons.default}
                </TimelineDot>
                {index < filteredEvents.length - 1 && <TimelineConnector />}
              </TimelineSeparator>
              
              <TimelineContent>
                <Paper elevation={3} sx={{ p: 2 }}>
                  <Typography variant="body1">
                    {event.description || (
                      event.event_type === 'person_appearance' 
                        ? `Person #${event.person_id} appears`
                        : event.event_type === 'person_disappearance'
                        ? `Person #${event.person_id} disappears`
                        : `Event: ${event.event_type}`
                    )}
                  </Typography>
                  
                  {/* Person details if available */}
                  {event.person_id !== undefined && personDetails[event.person_id] && (
                    <Box sx={{ mt: 1 }}>
                      <Button 
                        variant="outlined" 
                        size="small" 
                        href={`/cases/${caseId}/persons/${event.person_id}`}
                        sx={{ mt: 1 }}
                      >
                        View Person
                      </Button>
                      
                      {/* Position indicator if available */}
                      {event.position_3d && (
                        <Chip 
                          label="3D Position Available" 
                          size="small" 
                          color="primary" 
                          variant="outlined"
                          sx={{ ml: 1 }}
                        />
                      )}
                    </Box>
                  )}
                </Paper>
              </TimelineContent>
            </TimelineItem>
          ))}
        </Timeline>
      </Paper>
    </Box>
  );
}

export default TimelineView;