import React from 'react';
import { useNavigate } from 'react-router-dom';
import { IconButton, Tooltip } from '@mui/material';
import ChatIcon from '@mui/icons-material/Chat';

const ChatButton = ({ caseId }) => {
  const navigate = useNavigate();

  const handleChatClick = () => {
    navigate(`/case/${caseId}/chat`);
  };

  return (
    <Tooltip title="Chat with AI about this case">
      <IconButton
        color="primary"
        onClick={handleChatClick}
        sx={{
          transition: 'all 0.2s',
          '&:hover': {
            transform: 'scale(1.1)',
            bgcolor: 'rgba(37, 99, 235, 0.1)'
          }
        }}
      >
        <ChatIcon />
      </IconButton>
    </Tooltip>
  );
};

export default ChatButton;