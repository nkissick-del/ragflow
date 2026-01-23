import { cn } from '@/lib/utils';
import { Ellipsis } from 'lucide-react';
import React from 'react';
import { useTranslation } from 'react-i18next';
import { Button, ButtonProps } from './ui/button';

export const MoreButton = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, size, ...props }, ref) => {
    const { t } = useTranslation();
    return (
      <Button
        ref={ref}
        variant="ghost"
        size={size || 'icon'}
        aria-label={props['aria-label'] || t('common.action')}
        className={cn(
          'opacity-0 group-hover:opacity-100 focus-visible:opacity-100 transition-opacity size-3.5 bg-transparent group-hover:bg-transparent pointer-events-none group-hover:pointer-events-auto focus-visible:pointer-events-auto',
          className,
        )}
        {...props}
      >
        <Ellipsis />
      </Button>
    );
  },
);

MoreButton.displayName = 'MoreButton';
