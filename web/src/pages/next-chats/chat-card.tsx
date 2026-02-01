import { HomeCard } from '@/components/home-card';
import { MoreButton } from '@/components/more-button';
import { IDialog } from '@/interfaces/database/chat';
import { Routes } from '@/routes';
import { memo, useCallback } from 'react';
import { useNavigate } from 'react-router';
import { ChatDropdown } from './chat-dropdown';
import { useRenameChat } from './hooks/use-rename-chat';

export type IProps = {
  data: IDialog;
} & Pick<ReturnType<typeof useRenameChat>, 'showChatRenameModal'>;

// Memoized to prevent unnecessary re-renders when the parent list updates
export const ChatCard = memo(({ data, showChatRenameModal }: IProps) => {
  const navigate = useNavigate();

  const handleClick = useCallback(() => {
    navigate(`${Routes.Chat}/${data?.id}`);
  }, [navigate, data?.id]);

  return (
    <HomeCard
      data={{
        name: data.name,
        description: data.description,
        avatar: data.icon,
        update_time: data.update_time,
      }}
      moreDropdown={
        <ChatDropdown chat={data} showChatRenameModal={showChatRenameModal}>
          <MoreButton></MoreButton>
        </ChatDropdown>
      }
      onClick={handleClick}
    />
  );
});

ChatCard.displayName = 'ChatCard';
