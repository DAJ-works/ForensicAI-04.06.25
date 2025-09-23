import React, { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';
import { 
  Box, Typography, Paper, CircularProgress, IconButton,
  TextField, Button, Divider, Tooltip, Avatar
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import SendIcon from '@mui/icons-material/Send';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';

const CaseChat = () => {
  const { caseId } = useParams();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [caseInfo, setCaseInfo] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Fetch case info on component mount
  useEffect(() => {
    const fetchCaseInfo = async () => {
      try {
        const response = await axios.get(`/api/cases/${caseId}`);
        setCaseInfo(response.data);
        
        // Add initial system message
        setMessages([
          {
            role: 'assistant',
            content: `Welcome to Case Assistant for case #${caseId}. I can help answer questions about this case based on the video analysis data. What would you like to know?`
          }
        ]);
      } catch (error) {
        console.error('Error fetching case info:', error);
        setMessages([
          {
            role: 'assistant',
            content: `I couldn't load information for case #${caseId}. Please try again later or select a different case.`
          }
        ]);
      }
    };

    if (caseId) {
      fetchCaseInfo();
    }
    
    // Focus the input field
    setTimeout(() => {
      inputRef.current?.focus();
    }, 500);
  }, [caseId]);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message to chat
    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    
    // Clear input and show loading state
    setInput('');
    setIsLoading(true);

    try {
      // Send message to backend
      const response = await axios.post('/api/chat', {
        caseId: caseId,
        message: input
      });

      // Add response to chat
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: response.data.response 
      }]);
    } catch (error) {
      console.error('Error getting chat response:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error processing your request. Please try again later.' 
      }]);
    } finally {
      setIsLoading(false);
      // Focus input again after response
      inputRef.current?.focus();
    }
  };

  return (
    <Paper sx={{ 
      height: 'calc(100vh - 100px)', 
      display: 'flex', 
      flexDirection: 'column',
      borderRadius: 2,
      overflow: 'hidden'
    }}>
      {/* Chat Header */}
      <Box sx={{ 
        p: 2, 
        bgcolor: 'primary.main', 
        color: 'primary.contrastText',
        display: 'flex',
        alignItems: 'center'
      }}>
        <Tooltip title="Back to case">
          <IconButton 
            component={Link} 
            to={`/cases/${caseId}`} 
            sx={{ color: 'white', mr: 1 }}
            size="small"
          >
            <ArrowBackIcon />
          </IconButton>
        </Tooltip>
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Chat with AI Assistant
          </Typography>
          {caseInfo && (
            <Typography variant="body2" sx={{ opacity: 0.9 }}>
              Case #{caseId} {caseInfo.video_path && `• ${caseInfo.video_path.split('/').pop()}`}
            </Typography>
          )}
        </Box>
      </Box>
      
      {/* Chat Messages */}
      <Box sx={{ 
        flexGrow: 1, 
        overflowY: 'auto',
        p: 3,
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        bgcolor: theme => theme.palette.mode === 'light' ? '#f5f7fb' : '#111927'
      }}>
        {messages.map((message, index) => (
          <Box 
            key={index} 
            sx={{ 
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'flex-start',
              alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '80%'
            }}
          >
            {message.role === 'assistant' && (
              <Avatar 
                sx={{ 
                  bgcolor: 'primary.main', 
                  mr: 1.5,
                  width: 36,
                  height: 36
                }}
              >
                <SmartToyIcon sx={{ fontSize: 20 }} />
              </Avatar>
            )}
            
            <Paper 
              elevation={1}
              sx={{ 
                p: 2,
                borderRadius: 2,
                borderBottomRightRadius: message.role === 'user' ? 0 : 2,
                borderBottomLeftRadius: message.role === 'user' ? 2 : 0,
                bgcolor: message.role === 'user' ? 'primary.main' : 'background.paper',
                color: message.role === 'user' ? 'primary.contrastText' : 'text.primary',
              }}
            >
              <Typography variant="body1">
                {message.content}
              </Typography>
            </Paper>
            
            {message.role === 'user' && (
              <Avatar 
                sx={{ 
                  bgcolor: 'secondary.main', 
                  ml: 1.5,
                  width: 36,
                  height: 36
                }}
              >
                <PersonIcon sx={{ fontSize: 20 }} />
              </Avatar>
            )}
          </Box>
        ))}
        
        {isLoading && (
          <Box 
            sx={{ 
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'flex-start',
              alignSelf: 'flex-start',
              maxWidth: '80%'
            }}
          >
            <Avatar 
              sx={{ 
                bgcolor: 'primary.main', 
                mr: 1.5,
                width: 36,
                height: 36
              }}
            >
              <SmartToyIcon sx={{ fontSize: 20 }} />
            </Avatar>
            
            <Paper 
              elevation={1}
              sx={{ 
                p: 2,
                borderRadius: 2,
                borderBottomLeftRadius: 0,
                minWidth: 60,
                display: 'flex',
                justifyContent: 'center'
              }}
            >
              <CircularProgress size={20} thickness={5} />
            </Paper>
          </Box>
        )}
        
        <Box ref={messagesEndRef} />
      </Box>
      
      {/* Chat Input */}
      <Box 
        component="form" 
        onSubmit={handleSubmit}
        sx={{ 
          p: 2, 
          bgcolor: 'background.paper',
          borderTop: theme => `1px solid ${theme.palette.divider}`,
          display: 'flex',
          gap: 1
        }}
      >
        <TextField
          fullWidth
          placeholder="Ask a question about this case..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          variant="outlined"
          size="medium"
          disabled={isLoading}
          inputRef={inputRef}
          sx={{ 
            '& .MuiOutlinedInput-root': {
              borderRadius: 3,
              bgcolor: theme => theme.palette.mode === 'light' ? '#f5f7fb' : '#1a2436'
            }
          }}
        />
        <Button
          type="submit"
          variant="contained"
          color="primary"
          disabled={isLoading || !input.trim()}
          endIcon={<SendIcon />}
          sx={{ borderRadius: 2, px: 3 }}
        >
          Send
        </Button>
      </Box>
    </Paper>
  );
};

export default CaseChat;