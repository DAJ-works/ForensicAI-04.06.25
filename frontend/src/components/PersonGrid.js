import React from 'react';
import { Grid, Card, CardContent, Typography, CardMedia, Box, Chip, CardActionArea } from '@mui/material';
import { Link } from 'react-router-dom';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import PersonIcon from '@mui/icons-material/Person';
import LocationOnIcon from '@mui/icons-material/LocationOn';

function PersonGrid({ persons, caseId }) {
  if (!persons || persons.length === 0) {
    return (
      <Typography variant="body1" sx={{ mt: 2, mb: 4 }}>
        No persons detected in this case.
      </Typography>
    );
  }

  return (
    <Grid container spacing={3}>
      {persons.map((person) => (
        <Grid item key={person.id} xs={12} sm={6} md={4} lg={3}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardActionArea component={Link} to={`/cases/${caseId}/persons/${person.id}`}>
              {person.thumbnail ? (
                <CardMedia
                  component="img"
                  height="200"
                  image={person.thumbnail}
                  alt={`Person #${person.id}`}
                />
              ) : (
                <Box 
                  sx={{ 
                    height: 200, 
                    bgcolor: 'grey.300', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center' 
                  }}
                >
                  <PersonIcon sx={{ fontSize: 60, color: 'grey.500' }} />
                </Box>
              )}
              <CardContent>
                <Typography variant="h6" component="div" gutterBottom>
                  Person #{person.id}
                </Typography>
                
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <AccessTimeIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
                  <Typography variant="body2" color="text.secondary">
                    {person.appearances || person.metadata?.appearances || 0} appearances
                  </Typography>
                </Box>
                
                {person.position_3d && (
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <LocationOnIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
                    <Typography variant="body2" color="text.secondary">
                      3D position available
                    </Typography>
                  </Box>
                )}
                
                <Box sx={{ mt: 1 }}>
                  {person.metadata?.first_seen_time && (
                    <Chip 
                      label={`First seen: ${new Date(person.metadata.first_seen_time * 1000).toLocaleTimeString()}`} 
                      size="small" 
                      sx={{ mr: 1, mb: 1 }} 
                      variant="outlined"
                    />
                  )}
                  
                  {person.metadata?.attributes && (
                    <Box sx={{ mt: 1 }}>
                      {Object.entries(person.metadata.attributes).map(([attr, value]) => {
                        // For demonstration - in real app, would need proper attribute formatting
                        if (typeof value === 'object') return null;
                        return (
                          <Chip 
                            key={attr} 
                            label={`${attr}: ${value}`}
                            size="small" 
                            sx={{ mr: 1, mb: 1 }} 
                          />
                        );
                      })}
                    </Box>
                  )}
                </Box>
              </CardContent>
            </CardActionArea>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
}

export default PersonGrid;